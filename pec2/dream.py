"""Dream training: ES controller inside world-model rollouts.

Pipeline per ES generation:
  1. draw dream start states + property draws z (honest or poisoned)
  2. encode synthetic 8-step histories into latents [n, lat]
  3. replicate across population -> [P, n, lat]
  4. roll the latent with ZOH actions; decode; score with the reward model
  5. mean over episodes -> fitness [P]

Gate modes:
  honest : dreams condition on the true z
  noise  : + calibrated observation noise on decoded obs (2018-WM analog)
  gate   : + calibrated property jitter on z (uncertainty at the property level)
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from .envs import new_state, ACT_DIM, OBS_DIM, DT
from .es import ESNet, es_optimize, pack, unpack, run_mlp, flat_size


class RewardModel(nn.Module):
    """MLP: (decoded_obs, action) -> reward. Trained on real transitions."""
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], -1)).squeeze(-1)


def train_reward_model(rm, data, epochs=4, batch=1024, steps_per=20, lr=1e-3,
                       device="cpu"):
    """data: [Tn, N, ...] real-env dataset (minibatched)."""
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    obs, act, rew = (data["obs"].cpu(), data["act"].cpu(), data["rew"].cpu())
    Tn, N = obs.shape[0], obs.shape[1]
    tot, cnt = 0.0, 0
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        for _ in range(steps_per):
            ts = torch.randint(0, Tn, (batch,), generator=g)
            idx = torch.randint(0, N, (batch,), generator=g)
            o, a, r = obs[ts, idx].to(device), act[ts, idx].to(device), rew[ts, idx].to(device)
            loss = (rm(o, a) - r).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); cnt += 1
    return tot / max(1, cnt)


def _synth_start(n, device, gen):
    """Synthetic dream start consistent with the real env's initial distribution."""
    st = new_state(n, device, gen)
    return st


def dream_roll(wm, rm, z, n, T, cand, net, device, z_jitter=0.0, obs_noise=0.0,
               gen=None):
    """One batched dream rollout for the ES population.

    wm: world model with core.init_latent / step_lat / core.decode
    rm: reward model; z: [n, 5]; cand: [P, D]; net: ESNet(P=cand.shape[0])
    Returns fitness [P] (mean over n dream episodes).
    """
    P = cand.shape[0]
    if z_jitter > 0.0:
        zp = z + z_jitter * torch.randn(z.shape, generator=gen).to(z.device)
    else:
        zp = z
    st = _synth_start(n, device, gen)
    D = st["ee"].shape[-1]
    o0 = torch.cat([st["ee"], torch.zeros(n, D, device=device),
                    st["box"] - st["goal"], st["box"] - st["ee"]], -1)
    hist_o = o0[None].repeat(8, 1, 1)
    hist_a = torch.zeros(8, n, ACT_DIM, device=device)
    with torch.no_grad():
        x = wm.core.init_latent(hist_o, hist_a)            # [n, lat]
    x = x[None].expand(P, -1, -1).contiguous()             # [P, n, lat]
    obs = o0[None].expand(P, -1, -1).contiguous()          # [P, n, obs]
    zpP = zp[None].expand(P, -1, -1)
    ret = torch.zeros(P, n, device=device)
    with torch.no_grad():
        for t in range(T):
            a = net.forward(obs)                            # [P, n, 2]
            x = wm.step_lat(x, zpP, a, DT)
            obs2 = wm.core.decode(x, zpP)
            r = rm(obs2.reshape(P * n, -1), a.reshape(P * n, -1)).reshape(P, n)
            ret = ret + r
            if obs_noise > 0.0:
                obs2 = obs2 + obs_noise * torch.randn_like(obs2)
            obs = obs2
    return ret.mean(1).detach().cpu().numpy()


def train_controller_in_dreams(wm, rm, z, n, T, gens, pop, sigma, lr, seed=0,
                               log=None, mode="honest", lam=0.0, device="cpu"):
    """ES-train a controller inside the world model; returns (theta, hist).

    mode: honest | noise (obs-noise lam) | gate (property jitter lam)
    """
    sizes = [OBS_DIM, 64, 64, ACT_DIM]
    net = ESNet(sizes, P=pop, device=device, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.randn(net.D, generator=g).to(device) * 0.05

    def fitness(cand):
        return dream_roll(wm, rm, z, n, T, cand, net, device,
                          z_jitter=(lam if mode == "gate" else 0.0),
                          obs_noise=(lam if mode == "noise" else 0.0),
                          gen=g)

    theta, hist = es_optimize(net, theta, fitness, gens, pop, sigma, lr,
                              seed=seed, log=log)
    return theta, hist
