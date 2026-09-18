"""Property-conditioned latent world models (capacity-matched, clock is the
ONLY structural difference).

  LatentODEWorldModel   dx/dt = f(x, a, z); env step = RK4, fixed h=12.5 ms
                        (n_sub = round(dt/h) substeps, ZOH action)
  DiscreteTickWorldModel x_{t+1} = x_t + dt * f(x_t, a, z)   (single Euler step)

Both: GRU encoder over (obs, act) history -> latent x; decoder (x, z) -> obs.
z = hidden property vector, scaled by PROP_SCALE for conditioning.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .envs import PROP_SCALE, DT, OBS_DIM, ACT_DIM


def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class _Core(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, prop_dim=5, hid=64, lat=8):
        super().__init__()
        self.enc = nn.GRU(obs_dim + act_dim, hid, batch_first=True)
        self.proj = nn.Linear(hid, lat)
        self.f = mlp([lat + act_dim + prop_dim, 64, 64, lat])
        self.dec = mlp([lat + prop_dim, 64, 64, obs_dim])
        self.lat = lat

    def _zs(self, z):
        return z / PROP_SCALE.to(z.device)

    def init_latent(self, obs_seq, act_seq):
        """obs_seq/act_seq: [T, ..., D] (any leading dims) -> x [.., lat]."""
        shp = obs_seq.shape
        o = obs_seq.reshape(shp[0], -1, shp[-1])
        a = act_seq.reshape(shp[0], -1, ACT_DIM)
        out, _ = self.enc(torch.cat([o, a], -1))
        h = out[-1].reshape(*shp[1:-1], -1)
        return self.proj(h)

    def decode(self, x, z):
        return self.dec(torch.cat([x, self._zs(z)], -1))


class LatentODEWorldModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, prop_dim=5, hid=64,
                 lat=8, h=0.0125):
        super().__init__()
        self.core = _Core(obs_dim, act_dim, prop_dim, hid, lat)
        self.h = h

    def _f(self, x, a, z):
        return self.core.f(torch.cat([x, a, self.core._zs(z)], -1))

    def _rk4(self, x, a, z, h):
        k1 = self._f(x, a, z)
        k2 = self._f(x + 0.5 * h * k1, a, z)
        k3 = self._f(x + 0.5 * h * k2, a, z)
        k4 = self._f(x + h * k3, a, z)
        return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step_lat(self, x, z, a, dt):
        n_sub = max(1, int(round(dt / self.h)))
        h = dt / n_sub
        for _ in range(n_sub):
            x = self._rk4(x, a, z, h)
        return x

    def predict(self, obs_seq, act_seq, z, dt, a_next):
        """Teacher-forced next-step prediction: encode history, take one env
        step of latent dynamics under a_next, decode."""
        x = self.core.init_latent(obs_seq, act_seq)
        x = self.step_lat(x, z, a_next, dt)
        return self.core.decode(x, z)

    def roll_out(self, x0, z, actions, dt):
        """Open-loop: x0 [.., lat], actions [T, .., 2] -> decoded obs [T, .., obs]."""
        x = x0
        preds = []
        for t in range(actions.shape[0]):
            x = self.step_lat(x, z, actions[t], dt)
            preds.append(self.core.decode(x, z))
        return torch.stack(preds)


class DiscreteTickWorldModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, prop_dim=5, hid=64, lat=8):
        super().__init__()
        self.core = _Core(obs_dim, act_dim, prop_dim, hid, lat)

    def step_lat(self, x, z, a, dt):
        return x + dt * self.core.f(torch.cat([x, a, self.core._zs(z)], -1))

    def predict(self, obs_seq, act_seq, z, dt, a_next):
        x = self.core.init_latent(obs_seq, act_seq)
        x = self.step_lat(x, z, a_next, dt)
        return self.core.decode(x, z)

    def roll_out(self, x0, z, actions, dt):
        x = x0
        preds = []
        for t in range(actions.shape[0]):
            x = self.step_lat(x, z, actions[t], dt)
            preds.append(self.core.decode(x, z))
        return torch.stack(preds)


# ------------------------------------------------------------------ dataset
def collect_dataset(env, n_trajs, T, policy, device, gen=None):
    """Roll the batched env with `policy`.

    Property labels are tracked per-step (episodes resample props on reset):
      obs/act/rew: [n_trajs*T, N, D] (time-flattened across trajectory rolls)
      props: [n_trajs*T, N, 5] — the properties active at each step

    NOTE: within one trajectory roll the properties are constant except when a
    fault fires; the per-step tracking keeps labels honest across resets.
    """
    obs_l, act_l, rew_l, prop_l = [], [], [], []
    env.reset()
    # data collection is pure inference: no graph may leak into stored tensors
    # (a live graph here poisons every later backward with stale saved tensors)
    with torch.no_grad():
        for _ in range(n_trajs):
            oh, ah, rh = [], [], []
            obs = env.obs()
            for t in range(T):
                a = policy(obs)
                props_now = env.true_props()
                obs2, r, done = env.act(a)
                oh.append(obs.detach()); ah.append(a.detach()); rh.append(r.detach())
                prop_l.append(props_now.detach())
                obs = obs2
                if bool(done.all()):
                    break
            obs_l.append(torch.stack(oh))
            act_l.append(torch.stack(ah))
            rew_l.append(torch.stack(rh))
            env.reset()
    f = lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    Tn = obs_l[0].shape[0]
    return dict(obs=f(torch.stack(obs_l)), act=f(torch.stack(act_l)),
                rew=f(torch.stack(rew_l)),
                props=torch.stack(prop_l))  # [n_trajs*T, N, 5]


def make_windows(data, m, L=8, n_windows=8192, device="cpu", gen=None):
    """Mixed-clock windows: group m env steps (dt = m * DT, ZOH action).
    Returns hist_o [L, B, obs], hist_a [L, B, act], a_held [B, act] (first of
    the m held actions), z [B, 5], targets [B, obs] (obs after m steps)."""
    obs, act, props = data["obs"], data["act"], data["props"]
    Tn, N = obs.shape[0], obs.shape[1]
    # indices generated on the generator's device (cpu), moved to the DATA device
    ts = torch.randint(m * L, Tn - m, (n_windows,), generator=gen).to(obs.device)
    idx = torch.randint(0, N, (n_windows,), generator=gen).to(obs.device)
    ho, ha = [], []
    for j in range(L):
        tj = ts - m * (L - j)
        ho.append(obs[tj, idx])
        ha.append(act[tj, idx])
    a_held = act[ts - m, idx]                      # action held over the m steps
    tgt = obs[ts, idx]
    z = props[ts, idx]                             # properties active at target step
    return (torch.stack(ho).to(device), torch.stack(ha).to(device),
            a_held.to(device), z.to(device), tgt.to(device))


def train_wm(model, data, device, epochs=6, m_mix=(1,), lr=1e-3, batch=512,
             steps_per=8, seed=0):
    """Teacher-forced training on fixed-clock windows (default: env tick).

    E2's decisive design: BOTH models train at one clock (m=1) and are then
    evaluated across clocks/horizons. Any m_mix > 1 clock is an ablation
    ("mixed-clock training") that lets the fixed-tick model interpolate.

    epochs x len(m_mix) x steps_per gradient steps total.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tot, cnt = 0.0, 0
    for _ in range(epochs):
        for m in m_mix:
            for _ in range(steps_per):
                ho, ha, a_held, z, tgt = make_windows(data, m, n_windows=batch,
                                                      device=device, gen=g)
                pred = model.predict(ho, ha, z, m * DT, a_held)
                loss = (pred - tgt).square().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item(); cnt += 1
    return tot / max(1, cnt)


@torch.no_grad()
def eval_wm(model, data, m, k, device, n_windows=2048, seed=0):
    """k-step open-loop MSE at clock dt = m*DT on aligned fresh windows.

    History = 8 steps ending at ts-m (each history step is one env step at the
    SAME clock), the held action covers (ts-m, ts], so rollout step t predicts
    obs at ts + m*t... precisely: target[t] = obs[ts + m*t] for t = 1..k.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs, act, props = data["obs"], data["act"], data["props"]
    Tn, N = obs.shape[0], obs.shape[1]
    lo, hi = m * 8, Tn - m * k
    ts = torch.randint(lo, max(lo + 1, hi), (n_windows,), generator=g).to(obs.device)
    idx = torch.randint(0, N, (n_windows,), generator=g).to(obs.device)
    ho, ha = [], []
    for j in range(8):
        tj = ts - m * (8 - j)
        ho.append(obs[tj, idx]); ha.append(act[tj, idx])
    a_held = act[ts - m, idx]
    x = model.core.init_latent(torch.stack(ho).to(device),
                               torch.stack(ha).to(device))
    acts = a_held.to(device)[None].repeat(k, 1, 1)
    preds = model.roll_out(x, props[ts, idx].to(device), acts, m * DT)
    tot, cnt = 0.0, 0
    for t in range(k):
        tgt = obs[ts + m * t, idx].to(device)
        tot += (preds[t] - tgt).square().sum().item(); cnt += tgt.numel()
    return tot / max(1, cnt)
