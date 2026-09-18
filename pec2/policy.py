"""Policies, property encoder (teacher/student), RMA distillation student,
headless PPO actor-critic."""
from __future__ import annotations
import torch
import torch.nn as nn


def mlp(sizes, act=nn.Tanh):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class TinyPolicy(nn.Module):
    """Property-blind controller: obs -> action."""
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim])

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class PropertyEncoder(nn.Module):
    """Blind posterior over properties from a history of (obs, action).

    Trained by teacher forcing against privileged property labels.
    """
    def __init__(self, obs_dim, act_dim, hidden=128, prop_dim=5):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=False)
        self.head = nn.Linear(hidden, prop_dim)

    def forward(self, obs_seq, act_seq):
        # obs_seq/act_seq: [T, N, D]; fused cuDNN/cuDNN-free GRU over time
        x = self.inp(torch.cat([obs_seq, act_seq], -1))   # [T, N, hidden]
        out, _ = self.gru(x)                              # [T, N, hidden]
        return self.head(out)                             # [T, N, prop_dim]


class RMAStudent(nn.Module):
    """RMA stage-2 student (Kumar et al. 2021): GRU over (o_t, a_{t-1})
    regressing the NORMALIZED privileged vector the oracle-conditioned teacher
    consumed. Structurally the same readout as the PropertyEncoder but trained
    purely by distillation from on-policy teacher data.
    """
    def __init__(self, obs_dim, act_dim, hidden=64, prop_dim=5):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=False)
        self.head = nn.Linear(hidden, prop_dim)

    def forward(self, obs_seq, act_seq):
        x = self.inp(torch.cat([obs_seq, act_seq], -1))
        out, _ = self.gru(x)
        return self.head(out)                             # [T, N, prop_dim] (normalized)


class ActorCritic(nn.Module):
    """PPO actor-critic over latent state. Output head: normal action in [-1,1]^2."""
    def __init__(self, latent_dim, act_dim, hidden=64, det=False):
        super().__init__()
        self.body = mlp([latent_dim, hidden, hidden])
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.v = nn.Linear(hidden, 1)
        self.det = det

    def dist(self, feat):
        mu = torch.tanh(self.mu(self.body(feat)))
        return torch.distributions.Normal(mu, self.log_std.clamp(-2, 1).exp())

    def act(self, feat):
        d = self.dist(feat)
        a = d.mean if self.det else d.sample()
        return a.clamp(-1, 1)

    def evaluate(self, feat, action):
        d = self.dist(feat)
        return (d.log_prob(action).sum(-1),
                d.entropy().sum(-1),
                self.v(self.body(feat)).squeeze(-1))

    @torch.no_grad()
    def act_value(self, feat):
        d = self.dist(feat)
        a = (d.mean if self.det else d.sample()).clamp(-1, 1)
        return a, self.v(self.body(feat)).squeeze(-1)


def ppo_update(ac, buf, epochs=4, mb=256, lr=3e-4, clip=0.2, ent=0.01,
               target_kl=0.03):
    """buf: dict of stacked tensors: feat, act, logp, adv, ret."""
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    n = buf["feat"].shape[0]
    stats = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=buf["feat"].device)
        for i in range(0, n, mb):
            idx = perm[i:i + mb]
            logp, ent_t, val = ac.evaluate(buf["feat"][idx], buf["act"][idx])
            ratio = (logp - buf["logp"][idx]).exp()
            a = buf["adv"][idx]
            s1 = ratio * a
            s2 = ratio.clamp(1 - clip, 1 + clip) * a
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = 0.5 * (val - buf["ret"][idx]).square().mean()
            loss = pi_loss + 0.5 * v_loss - ent * ent_t.mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                approx_kl = (buf["logp"][idx] - logp).mean().abs().item()
            stats.append(approx_kl)
            if approx_kl > target_kl * 1.5:
                break
        if stats and stats[-1] > target_kl * 1.5:
            break
    return stats
