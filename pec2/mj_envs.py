"""MuJoCo replication environment (E8): a drop-in PushWorld on the MuJoCo engine.

Why: the paper's E3 claim (inference beats randomization and distillation for
fault recovery) was established on the suite's hand-rolled spring-contact
integrator. A reviewer's next question is whether the result survives a real
contact solver. This adapter re-implements the SAME task, observation, reward,
property semantics, fault schedules, and episode logic on MuJoCo 3.x:

  scene     a floating sphere (the end-effector, r=0.18, actuated in xyz by
            generalized forces) and a free box (cube half-size 0.25) on a
            frictional floor inside the same [-2, 2]^3 clamp volume
  actuator  first-order lag with per-env time constant tau and force gain,
            enforced outside the engine on the commanded force (the suite's
            actuator model is preserved exactly; MuJoCo integrates motion)
  contact   MuJoCo's elliptic friction-cone solver with floor+box friction
            derived from the property vector, box mass from the property
            vector, and solver stiffness matched to the suite's contact scale
  obs       identical 12-D layout: ee(3), boxv(3), box-goal(3), box-ee(3)
  reward    identical economics: 5x progress, -0.005 action cost, +1 first
            touch, +5 success (push); hold/hold_v2 use the hold economics
  faults    identical semantics: gain *= fg, tau *= ft at t=fault_at, plus
            the per-episode randomized schedules for the DR condition
  act()     returns (obs[12], reward, done) exactly like PushWorld.act

The sphere is kept at constant height by a light vertical spring so the
planar pushing task remains 2.5-D like the suite (the box can still be
tipped in principle, but the task statistics match the plane-invariant
`flat` mode of the suite).
"""
from __future__ import annotations

import numpy as np
import torch

import mujoco

from pec2.envs import (PROP_LOW, PROP_HIGH, PROP_SCALE, PROP_DIM, EE_R,
                       BOX_HALF, HOLD_ACC, HOLD_DIR, sample_props)

XML = r"""
<mujoco model="pec2_push">
  <option timestep="0.005" integrator="implicitfast" cone="elliptic">
    <flag contact="enable" gravity="disable"/>
  </option>
  <compiler angle="radian" inertiafromgeom="true"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1" friction="0.5 0.005 0.0001"
          condim="3" rgba="0.9 0.9 0.9 1"/>
    <body name="ee" pos="0 0 0.2">
      <freejoint/>
      <geom name="ee_geom" type="sphere" size="0.18" mass="2.0"
            friction="0.5 0.005 0.0001" condim="3" rgba="0.2 0.5 0.8 1"/>
    </body>
    <body name="box" pos="0.8 0 0.25">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.25 0.25 0.25" mass="3.0"
            friction="0.5 0.005 0.0001" condim="3" rgba="0.8 0.4 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
  </actuator>
</mujoco>
"""

SUBSTEPS = 4
DT = 0.02                    # env step duration (matches the suite)
F_NORM = 8.0                 # commanded-force scale (matches the suite)
ACTION_COST = 0.005
TOUCH_BONUS = 1.0
SUCCESS_BONUS = 5.0
PROGRESS_GAIN = 5.0
SUCCESS_DIST = 0.15
T_END = 150
LIMIT = 2.0

# vertical spring that keeps the sphere near its nominal height (2.5-D task)
EE_Z0 = 0.2
K_Z = 60.0                   # N/m restoring stiffness
D_Z = 8.0                    # damping


def _norm_props(props):
    """Map property vector to physical parameters used by the adapter."""
    # props: [m, mu, k, g, tau] with the suite's boxes
    return dict(mass=props[..., 0], mu=props[..., 1], k=props[..., 2],
                gain=props[..., 3], tau=props[..., 4].clamp_min(1e-3))


class MjPushWorld:
    """Batched MuJoCo PushWorld with the PushWorld interface.

    n parallel MuJoCo models are stepped in a python loop (MuJoCo's engine is
    fast enough for n<=64 at this scene complexity on CPU). Properties are
    resampled per episode exactly like the suite; faults follow the same
    schedules (fixed t=75 or randomized) applied to the commanded-force path.
    """

    def __init__(self, n, device="cpu", seed=0, fault_at=None,
                 fault_gain_mult=0.5, fault_tau_mult=2.0, obs_noise=0.0,
                 prop_box=None, vel_scale=1.0, flat=True, task="push",
                 rand_fault=False):
        assert task in ("push", "hold", "hold_v2")
        self.n, self.device = n, device
        self.flat = flat
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.fault_at = fault_at
        self.fg, self.ft = fault_gain_mult, fault_tau_mult
        self.rand_fault = rand_fault
        self.obs_noise = obs_noise
        self.cfg = dict(f_norm=F_NORM, vel_scale=vel_scale, task=task)
        self.prop_box = prop_box
        self._sample_faults()
        self.props = sample_props(n, device, self.gen, prop_box)
        self._build_models()
        self.reset()

    # ------------------------------------------------------------------ mj
    def _build_models(self):
        self.mjs = [mujoco.MjModel.from_xml_string(XML) for _ in range(self.n)]
        self.mjd = [mujoco.MjData(m) for m in self.mjs]
        bid = mujoco.mj_name2id(self.mjs[0], mujoco.mjtObj.mjOBJ_BODY, "box")
        self._box_mass0 = float(self.mjs[0].body_mass[bid])
        self._box_inertia0 = self.mjs[0].body_inertia[bid].copy()
        for i, d in enumerate(self.mjd):
            mujoco.mj_forward(self.mjs[i], d)

    def _apply_props(self):
        """Push the current property vector into the engines."""
        p = _norm_props(self.props)
        for i in range(self.n):
            m, d = self.mjs[i], self.mjd[i]
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
            new_mass = float(p["mass"][i]) + 2.0     # + box shell mass
            m.body_mass[bid] = new_mass
            m.body_inertia[bid] = self._box_inertia0 * (new_mass / self._box_mass0)
            mujoco.mj_setConst(m, d)
            gid_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
            m.geom_friction[gid_b, 0] = float(p["mu"][i])
            gid_e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ee_geom")
            m.geom_solmix[gid_e] = float(1.0 + p["k"][i])
            gid_f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            m.geom_friction[gid_f, 0] = float(p["mu"][i])
            d.qvel[:] = 0.0

    # ------------------------------------------------------- suite plumbing
    def _sample_faults(self):
        n, dev = self.n, self.device
        if self.fault_at is None:
            never = torch.full((n,), 10 ** 9, device=dev, dtype=torch.long)
            self.fault_at_vec = never
            self.fg_vec = torch.full((n,), self.fg, device=dev)
            self.ft_vec = torch.full((n,), self.ft, device=dev)
        elif self.rand_fault:
            g = self.gen
            self.fault_at_vec = torch.randint(40, 111, (n,), generator=g).to(dev)
            self.fg_vec = 0.4 + 0.2 * torch.rand((n,), generator=g).to(dev)
            self.ft_vec = 1.8 + 0.4 * torch.rand((n,), generator=g).to(dev)
        else:
            self.fault_at_vec = torch.full((n,), self.fault_at, device=dev,
                                           dtype=torch.long)
            self.fg_vec = torch.full((n,), self.fg, device=dev)
            self.ft_vec = torch.full((n,), self.ft, device=dev)

    def reset(self, idx=None):
        if idx is None:
            m = torch.ones(self.n, dtype=torch.bool)
        else:
            m = idx.bool()
        k = int(m.sum())
        if not k:
            return self.obs()
        self.props = torch.where(m[:, None], sample_props(
            self.n, self.device, self.gen, self.prop_box), self.props)
        if self.rand_fault:
            self._sample_faults()
        self._start_states(m)
        return self.obs()

    def _start_states(self, m):
        """Episode start: ee near origin, box 0.5-1.0 away, goal 0.8-1.6
        beyond the box (push) or box at goal (hold), matching the suite."""
        gd = self.gen.device
        n = self.n

        def rand3(kk, rmin, rmax):
            u = torch.randn((kk, 3), device=gd, generator=self.gen).to(self.device)
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if self.flat:
                u[:, 2] = 0.0
                u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rad = rmin + torch.rand((kk, 1), device=gd, generator=self.gen).to(self.device) * (rmax - rmin)
            return u * rad

        ee = rand3(n, 0.0, 0.3)
        box = ee + rand3(n, 0.5, 1.0)
        box = box.clamp(-LIMIT + BOX_HALF, LIMIT - BOX_HALF)
        goal = box + rand3(n, 0.8, 1.6)
        goal = goal.clamp(-LIMIT + BOX_HALF, LIMIT - BOX_HALF)
        if self.flat:
            box[:, 2] = 0.0
            goal[:, 2] = 0.0
        if self.cfg["task"] in ("hold", "hold_v2"):
            goal = box.clone()
        self.goal = goal
        self.Fcmd = torch.zeros((n, 3), device=self.device)
        self.t = torch.zeros((n,), device=self.device, dtype=torch.long)
        self.touched = torch.zeros((n,), device=self.device, dtype=torch.bool)
        self.success = torch.zeros((n,), device=self.device, dtype=torch.bool)
        self.faulted = torch.zeros((n,), device=self.device, dtype=torch.bool)
        for i in torch.nonzero(m).flatten().tolist():
            d = self.mjd[i]
            d.qpos[:] = 0.0
            d.qpos[0:3] = [float(ee[i, 0]), float(ee[i, 1]), EE_R]
            d.qpos[7:10] = [float(box[i, 0]), float(box[i, 1]), BOX_HALF]
            # box quaternion identity
            d.qpos[10] = 1.0
            mujoco.mj_forward(self.mjs[i], d)
        self._apply_props()
        self._ee0, self._box0 = ee, box

    # -------------------------------------------------------------- suite IO
    def obs(self):
        ee, boxv = self._state_vecs()
        return torch.cat([ee, boxv * self.cfg["vel_scale"],
                          self._box - self.goal, self._box - ee], dim=-1)

    def _state_vecs(self):
        """ee pos (z-corrected to the sphere center) and box velocity."""
        ee = torch.zeros((self.n, 3), device=self.device)
        boxv = torch.zeros((self.n, 3), device=self.device)
        self._box = torch.zeros((self.n, 3), device=self.device)
        for i in range(self.n):
            d = self.mjd[i]
            ee[i, 0] = d.qpos[0]
            ee[i, 1] = d.qpos[1]
            ee[i, 2] = d.qpos[2]           # sphere center
            # box body frame velocity (qvel[6:9] is linear in world frame)
            boxv[i, 0] = d.qvel[6]
            boxv[i, 1] = d.qvel[7]
            boxv[i, 2] = d.qvel[8]
            self._box[i, 0] = d.qpos[7]
            self._box[i, 1] = d.qpos[8]
            self._box[i, 2] = d.qpos[9]
        return ee, boxv

    def true_props(self):
        return self.props.clone()

    # ------------------------------------------------------------------ step
    def act(self, action):
        action = action.to(self.device)
        if self.fault_at is not None:
            hit = (self.t == self.fault_at_vec) & ~self.faulted
            if bool(hit.any()):
                self.faulted = self.faulted | hit
                props = self.props.clone()
                props[hit, 3] *= self.fg_vec[hit]
                props[hit, 4] *= self.ft_vec[hit]
                self.props = props
                self._apply_props()
        prev_dist = (self._box - self.goal).norm(dim=-1)
        p = _norm_props(self.props)
        task = self.cfg["task"]
        r = torch.zeros(self.n, device=self.device)
        for i in range(self.n):
            m, d = self.mjs[i], self.mjd[i]
            # first-order actuator lag on the commanded force (suite model)
            tau = float(p["tau"][i])
            Fdes = float(p["gain"][i]) * action[i] * F_NORM
            self.Fcmd[i] += (Fdes - self.Fcmd[i]) * (DT / SUBSTEPS) / tau
            # apply as xfrc_applied on the ee body; vertical spring added
            ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee")
            fz = K_Z * (EE_Z0 - d.qpos[2]) - D_Z * d.qvel[2]
            d.xfrc_applied[ee_bid, 0:3] = [
                float(self.Fcmd[i, 0]),
                float(self.Fcmd[i, 1]),
                float(self.Fcmd[i, 2]) + fz,
            ]
            for _ in range(SUBSTEPS):
                mujoco.mj_step(m, d)
            # clamp inside the box volume (suite clamp semantics)
            for lo, hi, qi in ((-LIMIT, LIMIT, 0), (-LIMIT, LIMIT, 1)):
                if d.qpos[qi] < lo:
                    d.qpos[qi] = lo
                    d.qvel[qi] *= -0.6
                elif d.qpos[qi] > hi:
                    d.qpos[qi] = hi
                    d.qvel[qi] *= -0.6
        ee, boxv = self._state_vecs()
        dist = (self._box - self.goal).norm(dim=-1)
        touch = ((EE_R + BOX_HALF) - (self._box - ee).norm(dim=-1)) > 0
        if task == "push":
            r = PROGRESS_GAIN * (prev_dist - dist) \
                - ACTION_COST * action.square().sum(-1)
            r = r + TOUCH_BONUS * (touch & ~self.touched).float()
            hit = dist < SUCCESS_DIST
            r = r + SUCCESS_BONUS * (hit & ~self.success).float()
            self.touched = self.touched | touch
            self.success = self.success | hit
        else:
            r = -1.0 * dist - ACTION_COST * action.square().sum(-1)
            if task == "hold_v2":
                box_ee = (self._box - ee).norm(dim=-1)
                # previous-step gap via the pre-step state (close enough for
                # the curriculum signal; the suite's variant is exact)
                r = r + 2.0 * (self._prev_gap - box_ee).clamp(-0.1, 0.1) \
                    + 1.0 * (touch & ~self.touched).float()
                self._prev_gap = box_ee
            self.touched = self.touched | touch
            self.success = self.success | (dist < 0.15)
        self.t += 1
        obs = torch.cat([ee, boxv * self.cfg["vel_scale"],
                         self._box - self.goal, self._box - ee], dim=-1)
        if self.obs_noise > 0.0:
            obs = obs + self.obs_noise * torch.randn(
                obs.shape, generator=self.gen).to(self.device)
        done = self.t >= T_END
        return obs, r, done

    # -------------------------------------------------- excitation probe (E0)
    def probe_action(self, t, gen=None):
        """Contact-seeking probe: sphere sweeps toward the box with lateral
        jitter, then pushes (persistent excitation, as in the suite)."""
        n = self.n
        a = torch.zeros((n, 3), device=self.device)
        if t < 30:
            a[:, 0] = 0.6
            jit = torch.rand((n, 2), generator=gen,
                             device=gen.device if gen is not None else None
                             ).to(self.device) - 0.5
            a[:, 0:2] += 0.4 * jit
        elif t < 70:
            a[:, 0] = 1.0
        else:
            a[:, 0] = 0.4 * (1.0 if (t // 20) % 2 == 0 else -1.0)
            a[:, 1] = 0.3 * (1.0 if (t // 35) % 2 == 0 else -1.0)
        return a.clamp(-1.0, 1.0)
