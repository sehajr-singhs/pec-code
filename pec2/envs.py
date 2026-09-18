"""Contact-rich 3-D pushing/holding with hidden object properties, lag
actuators, faults.

Physics (semi-implicit Euler, 2 substeps per env step, dt=0.02 s per step):
  end-effector (ee): sphere r=0.18, unit-ish mass, first-order actuator lag
      F_cmd    -> F_real with time constant tau (actuator property)
      F_real   = gain * F_cmd  (gain is an actuator property)
      drag on ee velocity
  contact: penetration p between ee sphere and box cube half-size 0.25
      F_c = stiff * k_base * p * n   (stiff is an object property, clamped)
      (point-contact along the center line; no torque - translational physics)
  box: cube, mass m (property), linear drag b = mu (property)

Hidden properties (never in the observation):
  mass in [1.5, 6.0], mu in [0.2, 3.0], stiff in [0.4, 2.5],
  gain in [0.4, 1.0], tau in [0.02, 0.12]

Observation (12-dim): ee(3), box velocity(3, scaled), box-goal(3), box-ee(3).
Faults: at a scheduled step, gain *= fault_gain_mult and tau *= fault_tau_mult
for the affected envs (mask-selectable).

`flat=True` restricts the world to the z=0 plane (2-D-compatible mode): all
positions start in-plane, the action z-component is masked out, and the plane
is invariant under the dynamics. Used for regression continuity and ablations.

Tasks (cfg["task"]):
  "push" (default): move the box to the goal; dense progress reward.
  "hold"  : a constant world-frame drift acceleration HOLD_ACC * HOLD_DIR
      pulls the box away from the goal (the box STARTS at the goal); reward
      = -|box-goal| each step. The ee must find and brace against the box on
      the updrift side; the required counter-force is m * a_g, so mass (and
      gain) are directly behaviorally relevant. With max sustainable contact
      force stiff*40 N (capped at 60) and a_g = 1.2 m/s^2, roughly 18% of the
      property box cannot be perfectly held (heavy box + weak actuator).
      NOTE: at small PPO budgets this reward has NO approach gradient (moving
      toward the box pays nothing until contact), so policies undertrain to
      below the passive-drift floor. "hold_v2" adds the push task's proven
      curriculum (approach shaping + first-touch bonus) without changing what
      the final competence is rewarded for.
  "hold_v2": hold + 2.0x box-ee approach shaping + +1 first-touch bonus.
"""

from __future__ import annotations

import torch

DT = 0.02
SUBSTEPS = 2
OBS_DIM = 12
ACT_DIM = 3
PROP_NAMES = ("mass", "mu", "stiff", "gain", "tau")
PROP_DIM = 5

PROP_LOW = torch.tensor([1.5, 0.2, 0.4, 0.4, 0.02])
PROP_HIGH = torch.tensor([6.0, 3.0, 2.5, 1.0, 0.12])
# conditioning scale for network inputs (roughly maps ranges to ~[-1,1])
PROP_SCALE = torch.tensor([6.0, 3.0, 2.5, 1.0, 0.12])

EE_R = 0.18
BOX_HALF = 0.25
LIMIT = 2.0
EE_MASS = 2.0
K_BASE = 40.0
F_MAX = 60.0
V_MAX = 8.0
DRAG_EE = 1.5
# hold task: constant world-frame drift acceleration (property-independent
# magnitude; the counter-force required scales with the hidden mass)
HOLD_ACC = 1.2
HOLD_DIR = torch.tensor([1.0, 0.5, 0.0]) / (1.0 ** 2 + 0.5 ** 2) ** 0.5


def sample_props(n, device, gen=None, box=None):
    """Uniform property sample. `box` = (low[5], high[5]) tensors to restrict."""
    lo = (PROP_LOW if box is None else box[0]).to(device)
    hi = (PROP_HIGH if box is None else box[1]).to(device)
    # generator lives on cpu (or device); sample there, then move
    gd = gen.device if gen is not None else torch.device("cpu")
    u = torch.rand((n, PROP_DIM), device=gd, generator=gen).to(device)
    return lo + u * (hi - lo)


def new_state(n, device, gen=None, with_faults=False, fault_at=0, flat=False,
              task="push"):
    """Randomized episode start: ee near origin, box 0.5-1.0 away, goal 0.8-1.6 from box.
    With flat=True all positions have z=0 (plane is invariant under the dynamics).
    task="hold": the box starts AT the goal and drifts away under the constant
    acceleration (the policy must brace against it)."""
    gd = gen.device if gen is not None else torch.device("cpu")

    def rand3(n, rmin, rmax):
        u = torch.randn((n, 3), device=gd, generator=gen).to(device)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        if flat:
            u[:, 2] = 0.0
            u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rad = rmin + torch.rand((n, 1), device=gd, generator=gen).to(device) * (rmax - rmin)
        return u * rad

    ee = rand3(n, 0.0, 0.3)
    box = ee + rand3(n, 0.5, 1.0)
    box = box.clamp(-LIMIT + BOX_HALF, LIMIT - BOX_HALF)
    goal = box + rand3(n, 0.8, 1.6)
    goal = goal.clamp(-LIMIT + BOX_HALF, LIMIT - BOX_HALF)
    if flat:
        box[:, 2] = 0.0
        goal[:, 2] = 0.0
    if task in ("hold", "hold_v2"):
        goal = box.clone()
    D = 3
    st = dict(
        ee=ee, eev=torch.zeros((n, D), device=device),
        box=box, boxv=torch.zeros((n, D), device=device),
        goal=goal, Fcmd=torch.zeros((n, D), device=device),
        t=torch.zeros((n,), device=device, dtype=torch.long),
        touched=torch.zeros((n,), device=device, dtype=torch.bool),
        success=torch.zeros((n,), device=device, dtype=torch.bool),
        faulted=(torch.zeros((n,), device=device, dtype=torch.bool) if with_faults
                 else None),
    )
    return st


def _integrate(st, props, action, cfg):
    """One semi-implicit Euler substep of the full contact dynamics."""
    gain, tau = props[:, 3:4], props[:, 4:5].clamp_min(1e-3)
    mass, mu, stiff = props[:, 0:1], props[:, 1:2], props[:, 2:3]
    a = action.clamp(-1.0, 1.0)
    Fdes = gain * a * cfg["f_norm"]
    # actuator lag toward commanded force
    Fcmd = st["Fcmd"] + (Fdes - st["Fcmd"]) * (DT / SUBSTEPS) / tau
    Free = Fcmd - DRAG_EE * st["eev"]
    # contact
    d = st["box"] - st["ee"]
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pen = (EE_R + BOX_HALF) - dist
    n = d / dist
    Fc = (stiff * K_BASE * pen.clamp_min(0.0)).clamp(max=F_MAX) * n
    # hold task: constant drift acceleration on the box (world frame)
    drift = (HOLD_ACC * HOLD_DIR.to(st["box"].device)[None] * (DT / SUBSTEPS)
             if cfg.get("task") == "hold" else 0.0)
    # integrate velocities then positions
    eev = st["eev"] + (Free - Fc / EE_MASS) * (DT / SUBSTEPS)
    boxv = st["boxv"] + (Fc / mass - mu * st["boxv"]) * (DT / SUBSTEPS) + drift
    eev = eev.clamp(-V_MAX, V_MAX)
    boxv = boxv.clamp(-V_MAX, V_MAX)
    ee = st["ee"] + eev * (DT / SUBSTEPS)
    box = st["box"] + boxv * (DT / SUBSTEPS)
    ee = ee.clamp(-LIMIT + EE_R, LIMIT - EE_R)
    boxv = torch.where((box.abs() > LIMIT - BOX_HALF), -0.6 * boxv, boxv)
    box = box.clamp(-LIMIT + BOX_HALF, LIMIT - BOX_HALF)
    return dict(st, ee=ee, eev=eev, box=box, boxv=boxv, Fcmd=Fcmd)


def step(st, props, action, cfg=None, obs_noise=0.0, gen=None, flat=False):
    """Advance one env step; returns (new_state, obs[N,12], reward[N])."""
    cfg = cfg or {}
    task = cfg.get("task", "push")
    if flat:
        action = action.clone()
        action[:, 2] = 0.0
    for _ in range(SUBSTEPS):
        st = _integrate(st, props, action, cfg)
    t1 = st["t"] + 1
    st = dict(st, t=t1)
    dist = (st["box"] - st["goal"]).norm(dim=-1)
    if task in ("hold", "hold_v2"):
        # hold: be close to the goal every step; drift makes distance grow
        # without an active bracing policy
        r = -1.0 * dist - 0.005 * action.square().sum(-1)
        if task == "hold_v2":
            # approach curriculum (identical in spirit to push's first-touch):
            # pay for closing the box-ee gap, then for first contact
            box_ee = (st["box"] - st["ee"]).norm(dim=-1)
            prev_box_ee = (st["box"] - st["boxv"] * DT - st["ee"]).norm(dim=-1)
            touch = ((EE_R + BOX_HALF) - box_ee) > 0
            r = r + 2.0 * (prev_box_ee - box_ee) \
                + 1.0 * (touch & ~st["touched"]).float()
            st = dict(st, touched=st["touched"] | touch)
        st = dict(st, success=st["success"] | (dist < 0.15))
    else:
        prev_dist = (st["box"] - st["boxv"] * DT - st["goal"]).norm(dim=-1)
        # dense shaping, rebalanced for 3-D: progress must dominate the action
        # cost or the optimal policy freezes (progress ~0.005 m/step for decent
        # pushing); first-touch bonus gives PPO a two-stage curriculum
        # (learn contact, then push) without changing what is rewarded.
        touch = ((EE_R + BOX_HALF) - (st["box"] - st["ee"]).norm(dim=-1)) > 0
        r = 5.0 * (prev_dist - dist) - 0.005 * action.square().sum(-1)
        r = r + 1.0 * (touch & ~st["touched"]).float()
        hit = dist < 0.15
        r = r + 5.0 * (hit & ~st["success"]).float()
        st = dict(st, touched=st["touched"] | touch, success=st["success"] | hit)
    vel_scale = cfg.get("vel_scale", 1.0)
    # obs: ee(3), boxv(3), box-goal(3), box-ee(3)  -- goal IS observable
    obs = torch.cat([st["ee"], st["boxv"] * vel_scale,
                     st["box"] - st["goal"], st["box"] - st["ee"]], dim=-1)
    if obs_noise > 0.0:
        obs = obs + obs_noise * torch.randn(obs.shape, generator=gen).to(obs.device)
    return st, obs, r


class PushWorld:
    """Batched env with property resampling per episode and an optional fault
    schedule applied at `fault_at` (gain *= fg, tau *= ft).

    rand_fault=True replaces the fixed schedule with a per-episode RANDOM one:
    fault time ~ U{40..110}, gain mult ~ U[0.4, 0.6], tau mult ~ U[1.8, 2.2]
    (covers the canonical deployment fault t=75, mults 0.5/2.0 without
    memorizing its timing). Used by the domain-randomization baseline.
    """

    def __init__(self, n, device="cpu", seed=0, fault_at=None,
                 fault_gain_mult=0.5, fault_tau_mult=2.0, obs_noise=0.0,
                 prop_box=None, vel_scale=1.0, flat=False, task="push",
                 rand_fault=False):
        self.n, self.device = n, device
        self.flat = flat
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.fault_at = fault_at
        self.fg, self.ft = fault_gain_mult, fault_tau_mult
        self.rand_fault = rand_fault
        self.obs_noise = obs_noise
        self.cfg = dict(f_norm=8.0, vel_scale=vel_scale, task=task)
        self.prop_box = prop_box
        self._sample_faults()
        self.props = sample_props(n, device, self.gen, prop_box)
        self.st = new_state(n, device, self.gen,
                            with_faults=fault_at is not None,
                            flat=self.flat, task=task)

    def _sample_faults(self):
        """Per-env fault schedule (time, gain mult, tau mult) as tensors."""
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
            self.props = sample_props(self.n, self.device, self.gen, self.prop_box)
            if self.rand_fault:
                self._sample_faults()
            self.st = new_state(self.n, self.device, self.gen,
                                with_faults=self.fault_at is not None,
                                flat=self.flat, task=self.cfg["task"])
        else:
            m = idx
            k = int(m.sum())
            if k:
                self.props = torch.where(m[:, None], sample_props(
                    self.n, self.device, self.gen, self.prop_box), self.props)
                if self.rand_fault:
                    keep = {kk: vv[m].clone() for kk, vv in self.__dict__.items()
                            if kk.endswith("_vec")}
                    self._sample_faults()
                    for kk, vv in keep.items():
                        cur = getattr(self, kk)
                        setattr(self, kk, torch.where(m, cur, vv))
                fresh = new_state(self.n, self.device, self.gen,
                                  with_faults=self.fault_at is not None,
                                  flat=self.flat, task=self.cfg["task"])
                for key, v in self.st.items():
                    if v is None:
                        continue
                    upd = torch.where(m[:, None] if v.dim() > 1 else m,
                                      fresh[key], v)
                    self.st[key] = upd
        return self.obs()

    def obs(self):
        return torch.cat([self.st["ee"], self.st["boxv"] * self.cfg["vel_scale"],
                          self.st["box"] - self.st["goal"],
                          self.st["box"] - self.st["ee"]], dim=-1)

    def act(self, action):
        if self.fault_at is not None:
            hit = (self.st["t"] == self.fault_at_vec) & ~self.st["faulted"]
            if bool(hit.any()):
                self.st["faulted"] = self.st["faulted"] | hit
                props = self.props.clone()
                props[hit, 3] *= self.fg_vec[hit]
                props[hit, 4] *= self.ft_vec[hit]
                self.props = props
        self.st, obs, r = step(self.st, self.props, action, self.cfg,
                               self.obs_noise, self.gen, flat=self.flat)
        done = self.st["t"] >= 150
        return obs, r, done

    def true_props(self):
        return self.props.clone()
