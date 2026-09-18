"""Experiments E0-E4. Each returns a JSON-ready dict.

E0  estimator: teacher-forced GRU posterior vs privileged labels (NRMSE vs mean-predictor)
E1  dream gate: honest | obs-noise | property-jitter gate, trained on poisoned dreams
E2  clock sweep: latent-ODE vs fixed-tick (capacity-matched), horizon + clock transfer
E3  fault adaptation: PPO-trained blind vs z-conditioned (online posterior) vs oracle
E4  cross-modal transfer: narrow-box training, stratified evaluation (central/edge/2-edge)
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn

from .envs import (PushWorld, sample_props, new_state, step, PROP_LOW, PROP_HIGH,
                   PROP_SCALE, PROP_DIM, DT, OBS_DIM, ACT_DIM)
from .policy import (PropertyEncoder, TinyPolicy, ActorCritic, ppo_update,
                     RMAStudent)
from .wm import (LatentODEWorldModel, DiscreteTickWorldModel, collect_dataset,
                 train_wm, eval_wm)
from .dream import (RewardModel, train_reward_model, train_controller_in_dreams,
                    dream_roll)
from .es import ESNet
from .statx import boot_ci, paired_report

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def _single_net(theta, sizes, device):
    from .es import ESNet
    return ESNet.from_theta(theta.to(device), sizes, device=device)


# ------------------------------------------------------------------- shared
def _probe_policy(o, gen, amp=0.6, noise=0.5):
    """Contact-seeking excitation probe (system-ID): steer the ee into the box
    and keep poking. Uses ONLY the observation (box-ee direction), so the
    encoder still has to infer properties from the response - this is the
    classic persistent-excitation requirement, without it mass/mu/stiff are
    unidentifiable because contact never happens.
    """
    d = o[:, -3:]                                  # box - ee (last 3 dims)
    u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    eps = torch.randn(u.shape, generator=gen).to(o.device)
    return (amp * u + noise * eps).clamp(-1, 1)


def _train_encoder(env, enc, device, seed, epochs=8, L=32, n_win=512,
                   steps_per=60, T=150, n_trajs=6, pol=None, n_cycles=3,
                   prop_scale=None, val_batch=512):
    """Teacher-forced property-encoder training on real env data.

    Targets are range-normalized to ~[-1, 1]. Data is collected with a
    contact-seeking excitation probe (see _probe_policy): without persistent
    excitation a smooth policy never makes contact and mass/mu/stiff are
    unidentifiable.

    Anti-memorization protocol (all three are needed):
      1. validation windows come from held-out EPISODES (window-level splits
         leak: adjacent windows share 31/32 timesteps),
      2. best-val checkpointing with restore at the end,
      3. the pool is refreshed with FRESH episodes (newly resampled
         properties) every `steps_cycle` gradient steps.
    """
    g_pol = torch.Generator(device="cpu").manual_seed(seed + 555)
    scale = PROP_SCALE if prop_scale is None else prop_scale

    def probe(o):
        return _probe_policy(o, g_pol)

    # keep datasets CPU-resident: window indexing happens on CPU, only the
    # minibatch fed to the encoder is moved to `device`
    lo, hi = PROP_LOW, PROP_HIGH                   # CPU: pool + targets live here
    span = (hi - lo)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def collect_pool():
        with torch.no_grad():
            data = collect_dataset(env, n_trajs=n_trajs, T=T, policy=probe,
                                   device=device)
        obs = data["obs"].cpu()
        act = data["act"].cpu()
        props = data["props"].cpu()
        Tn, N = obs.shape[0], obs.shape[1]
        # normalize each target by its own conditioning scale (tau needs 0.12,
        # not the shared 1.0); targets land in ~[-1, 1]
        z01 = ((props - lo) / scale) * 2 - 1
        perm = torch.randperm(N, generator=g)
        n_val_e = max(2, N // 8)
        val_e, tr_e = perm[:n_val_e], perm[n_val_e:]
        pool = min(32768, N * (Tn - L))
        ts = torch.randint(L, Tn, (pool,), generator=g)
        idx = torch.randint(0, N, (pool,), generator=g)
        is_val = torch.isin(idx, val_e)
        HO = torch.empty(int((~is_val).sum()), L, OBS_DIM)
        HA = torch.empty(int((~is_val).sum()), L, ACT_DIM)
        for j in range(L):
            HO[:, j] = obs[ts[~is_val] - L + 1 + j, idx[~is_val]].cpu()
            HA[:, j] = act[ts[~is_val] - L + 1 + j, idx[~is_val]].cpu()
        Z = z01[ts[~is_val], idx[~is_val]].cpu()
        HOv = torch.stack([obs[ts[is_val] - L + 1 + j, idx[is_val]]
                           for j in range(L)], dim=1)
        HAv = torch.stack([act[ts[is_val] - L + 1 + j, idx[is_val]]
                           for j in range(L)], dim=1)
        Zv = z01[ts[is_val], idx[is_val]].cpu()
        return HO, HA, Z, HOv, HAv, Zv

    # weight decay + input jitter + episode-held-out early stopping:
    # without them the encoder memorizes the pool (train loss keeps dropping
    # while held-out NRMSE rises)
    opt = torch.optim.Adam(enc.parameters(), 1e-3, weight_decay=1e-4)
    total_steps = epochs * steps_per
    steps_cycle = max(40, total_steps // n_cycles)
    best_val, best_state = float("inf"), None
    done = 0
    while done < total_steps:
        HO, HA, Z, HOv, HAv, Zv = collect_pool()
        n_tr, n_va = HO.shape[0], HOv.shape[0]
        mb = min(n_win, n_tr)
        for _ in range(min(steps_cycle, total_steps - done)):
            b = torch.randint(0, n_tr, (mb,), generator=g)
            # encoder expects [T, N, D]; dim0=time, so feed [L, mb, D], [-1]
            obs_mb = HO[b].transpose(0, 1).to(device)
            obs_mb = obs_mb + 0.01 * torch.randn(obs_mb.shape, device=device)
            pred = enc(obs_mb, HA[b].transpose(0, 1).to(device))[-1]
            loss = ((pred - Z[b].to(device)) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            done += 1
            if done % 20 == 0:
                with torch.no_grad():
                    j = torch.randint(0, n_va, (512,), generator=g)
                    pv = enc(HOv[j].transpose(0, 1).to(device),
                             HAv[j].transpose(0, 1).to(device))[-1]
                    vloss = float(((pv - Zv[j].to(device)) ** 2).mean())
                if vloss < best_val:
                    best_val = vloss
                    best_state = {k: t.detach().clone()
                                  for k, t in enc.state_dict().items()}
        if done >= total_steps:
            break
    if best_state is not None:
        enc.load_state_dict(best_state)
    return dict(final_loss=float(best_val), span=span.cpu(), lo=lo.cpu(),
                scale=scale.cpu())


def _encode_norm(enc, hist_o, hist_a, lo, span, device):
    """Encoder output (in [-1,1] normalized space) -> physical property estimate."""
    with torch.no_grad():
        zn = enc(hist_o, hist_a)[-1]
    return lo.to(device) + (zn + 1) / 2 * span.to(device)


# ------------------------------------------------------------------- E0
def exp0_estimator(seeds=range(5), n=256, enc_epochs=12, device=DEV):
    """Encoder NRMSE vs a mean-predictor baseline (the honest bar)."""
    per_seed = []
    for s in seeds:
        set_seed(1000 + s)
        env = PushWorld(n=n, device=device, seed=2000 + s)
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        info = _train_encoder(env, enc, device, seed=s, epochs=enc_epochs)
        lo, span = info["lo"].to(device), info["span"].to(device)
        # held-out eval (same excitation probe protocol - system-ID standard)
        env2 = PushWorld(n=n, device=device, seed=3000 + s)
        g2 = torch.Generator(device="cpu").manual_seed(s + 777)

        def probe2(o):
            return _probe_policy(o, g2)

        with torch.no_grad():
            data2 = collect_dataset(env2, n_trajs=4, T=150, policy=probe2, device=device)
        obs2, act2, props2 = (data2["obs"].cpu(), data2["act"].cpu(),
                              data2["props"].cpu())
        L = 32
        g = torch.Generator(device="cpu").manual_seed(s)
        ts = torch.randint(L, obs2.shape[0], (2048,), generator=g)
        idx = torch.randint(0, n, (2048,), generator=g)
        ho = torch.stack([obs2[ts - L + 1 + j, idx] for j in range(L)]).to(device)
        ha = torch.stack([act2[ts - L + 1 + j, idx] for j in range(L)]).to(device)
        z = props2[ts, idx].to(device)
        with torch.no_grad():
            zn = enc(ho, ha)[-1]
        scale = info["scale"].to(device)
        pred = lo + (zn + 1) / 2 * scale
        rmse = (pred - z).square().mean(0).sqrt()
        # mean-predictor baseline on the same eval windows
        mean_pred = props2.reshape(-1, PROP_DIM).mean(0).to(device)
        rmse_mean = (mean_pred - z).square().mean(0).sqrt()
        nrmse = (rmse / span).mean()
        nrmse_mean = (rmse_mean / span).mean()
        per_seed.append(dict(seed=s, nrmse=float(nrmse),
                             nrmse_mean=float(nrmse_mean),
                             per_prop=[float(x) for x in (rmse / span)]))
    arr = np.array([p["nrmse"] for p in per_seed])
    arrm = np.array([p["nrmse_mean"] for p in per_seed])
    return dict(name="estimator", n_seeds=len(per_seed), per_seed=per_seed,
                mean_nrmse=float(arr.mean()), ci=boot_ci(arr),
                mean_predictor_nrmse=float(arrm.mean()), ci_mean=boot_ci(arrm))


# ------------------------------------------------------------------- E1
def exp1_gate_vs_noise(seeds=range(6), gens=120, pop=64, T=40, n_dream=64,
                       device=DEV, log=print):
    """Honest | obs-noise | property-jitter gate, all trained on poisoned dreams
    (phantom actuator gain 1.6x), evaluated on the REAL env."""
    log = log or (lambda *a, **k: None)
    per_seed = []
    for s in seeds:
        set_seed(100 + s)
        env = PushWorld(n=256, device=device, seed=400 + s)
        pol = TinyPolicy(OBS_DIM, ACT_DIM).to(device)
        data = collect_dataset(env, n_trajs=6, T=150, policy=pol, device=device)
        wm = LatentODEWorldModel().to(device)
        wl = train_wm(wm, data, device, epochs=6, seed=s)
        rm = RewardModel().to(device)
        rl = train_reward_model(rm, data, device=device)

        z = sample_props(n_dream, device,
                         torch.Generator(device="cpu").manual_seed(500 + s))
        poison = dict(gain=1.6)
        zp = z.clone(); zp[:, 3] *= poison["gain"]     # poisoned conditioning

        # contrastive gate calibration: probe ES in poisoned vs honest dreams
        theta_p, _ = train_controller_in_dreams(
            wm, rm, zp, n_dream, T, gens=25, pop=32, sigma=0.1, lr=0.05,
            seed=901 + s, mode="honest", device=device)
        theta_h, _ = train_controller_in_dreams(
            wm, rm, z, n_dream, T, gens=25, pop=32, sigma=0.1, lr=0.05,
            seed=900 + s, mode="honest", device=device)
        sizes = [OBS_DIM, 64, 64, ACT_DIM]
        net1 = ESNet(sizes, P=1, device=device, seed=0)
        g = torch.Generator(device="cpu").manual_seed(600 + s)
        gap_p = float(dream_roll(wm, rm, z, n_dream, T, theta_p[None], net1,
                                 device, gen=g).mean())
        gap_h = float(dream_roll(wm, rm, z, n_dream, T, theta_h[None], net1,
                                 device, gen=g).mean())
        exploit = max(0.0, gap_p - gap_h)
        # calibrated uncertainty: scale jitter with the measured exploit gap
        lam = float(np.clip(0.15 * exploit / max(1e-3, abs(gap_h) + 0.1), 0.0, 0.4))

        out = dict(seed=s, wm_loss=wl, rm_loss=rl, exploit=exploit, lam=lam,
                   gap_h=gap_h, gap_p=gap_p)
        for mode, mlam in (("honest", 0.0), ("noise", max(lam, 0.05)),
                           ("gate", lam)):
            th, hist = train_controller_in_dreams(
                wm, rm, zp, n_dream, T, gens=gens, pop=pop, sigma=0.1,
                lr=0.05, seed=700 + s + (0 if mode == "honest" else
                                         (1 if mode == "noise" else 2)),
                mode=mode, lam=mlam, device=device,
                log=(log if s == 0 else None))
            real = _real_eval(th, sizes, n=256, T=150, device=device,
                              seed=800 + s)
            out[mode] = real
        per_seed.append(out)
        if log:
            log(f"  E1 seed {s}: honest {out['honest']:+.3f} noise {out['noise']:+.3f} "
                f"gate {out['gate']:+.3f} (exploit {exploit:.3f} lam {lam:.3f})")
    stats = {
        "gate_vs_noise": paired_report([p["gate"] for p in per_seed],
                                       [p["noise"] for p in per_seed]),
        "gate_vs_honest": paired_report([p["gate"] for p in per_seed],
                                        [p["honest"] for p in per_seed]),
        "noise_vs_honest": paired_report([p["noise"] for p in per_seed],
                                         [p["honest"] for p in per_seed]),
    }
    return dict(name="gate_vs_noise", n_seeds=len(per_seed), per_seed=per_seed,
                stats=stats)


def _real_eval(theta, sizes, n, T, device, seed):
    """Flat-theta controller on the real batched env -> mean return."""
    f = _single_net(theta, sizes, device)
    env = PushWorld(n=n, device=device, seed=seed)
    env.reset()
    ret = torch.zeros(n, device=device)
    obs = env.obs()
    for t in range(T):
        a = f(obs)
        obs, r, done = env.act(a)
        ret += r
        if bool(done.all()):
            break
    return float(ret.mean().item())


# ------------------------------------------------------------------- E2
def exp2_clock_sweep(dts=(1, 2, 4, 8), ks=(1, 2, 4, 8), seeds=range(3),
                     device=DEV, log=print):
    """k-step open-loop MSE at clock dt = m*DT; both models capacity-matched
    and trained at the env tick only (m=1), then evaluated across clocks."""
    log = log or (lambda *a, **k: None)
    agg = {"latent_ode": {}, "discrete_tick": {}}
    train_mse = {"latent_ode": [], "discrete_tick": []}
    for s in seeds:
        set_seed(42 + s)
        env = PushWorld(n=256, device=device, seed=42 + s)
        pol = TinyPolicy(OBS_DIM, ACT_DIM).to(device)
        data = collect_dataset(env, n_trajs=6, T=150, policy=pol, device=device)
        wm_ode = LatentODEWorldModel().to(device)
        wm_tick = DiscreteTickWorldModel().to(device)
        train_mse["latent_ode"].append(train_wm(wm_ode, data, device, epochs=6, seed=s))
        train_mse["discrete_tick"].append(train_wm(wm_tick, data, device, epochs=6, seed=s))
        for m in dts:
            for k in ks:
                mo = eval_wm(wm_ode, data, m, k, device, seed=1000 + 7 * m + k)
                mk = eval_wm(wm_tick, data, m, k, device, seed=1000 + 7 * m + k)
                agg["latent_ode"].setdefault(str(m), {}).setdefault(str(k), []).append(mo)
                agg["discrete_tick"].setdefault(str(m), {}).setdefault(str(k), []).append(mk)
        log(f"  E2 seed {s} done")
    mse = {mdl: {m: {k: float(np.mean(v)) for k, v in d.items()}
                 for m, d in dd.items()} for mdl, dd in agg.items()}
    return dict(name="clock_sweep", dts=list(dts), ks=list(ks), mse=mse,
                train_mse={k: [float(x) for x in v] for k, v in train_mse.items()})


# ------------------------------------------------------------------- PPO infra
def _run_ppo_env(env, ac, enc, device, T=150, gamma=0.99, lam_gae=0.95,
                 zcond=False, oracle=False):
    """Collect one batch of on-policy trajectories with optional z-conditioning.
    Returns (buf, episode_ret[n])."""
    n = env.n
    obs = env.obs()
    hist_o, hist_a = None, None
    feats, acts, logps, vals, rews, dones = [], [], [], [], [], []
    ret_ep = torch.zeros(n, device=device)
    for t in range(T):
        if zcond:
            if hist_o is None:
                hist_o = obs[None].repeat(8, 1, 1)
                hist_a = torch.zeros(8, n, ACT_DIM, device=device)
            else:
                hist_o = torch.cat([hist_o[1:], obs[None]])
                hist_a = torch.cat([hist_a[1:], acts[-1][None]])
            with torch.no_grad():
                if oracle:
                    zz = env.true_props()
                else:
                    lo = PROP_LOW.to(device); hi = PROP_HIGH.to(device)
                    scale = (hi - lo) / PROP_SCALE.to(device)
                    zn = enc(hist_o, hist_a)[-1]
                    zz = lo + (zn + 1) / 2 * scale
            feat = torch.cat([obs, zz / PROP_SCALE.to(device)], -1)
        else:
            feat = obs
        with torch.no_grad():
            a, val_t = ac.act_value(feat)
            logp_t = ac.dist(feat).log_prob(a).sum(-1)
        feats.append(feat); acts.append(a); logps.append(logp_t)
        vals.append(val_t)
        obs, r, done = env.act(a)
        rews.append(r); dones.append(done)
        ret_ep += r
        if bool(done.all()):
            break
    Tn = len(rews)
    rews = torch.stack(rews)                    # [Tn, n]
    dones = torch.stack(dones)                  # [Tn, n]
    vals = torch.stack(vals)                    # [Tn, n]
    with torch.no_grad():
        adv = torch.zeros_like(rews)
        lastgae = torch.zeros(n, device=device)
        for t in reversed(range(Tn)):
            nextv = vals[t + 1] if t + 1 < Tn else torch.zeros(n, device=device)
            mask = (~dones[t]).float()
            delta = rews[t] + gamma * nextv * mask - vals[t]
            lastgae = delta + gamma * lam_gae * mask * lastgae
            adv[t] = lastgae
        ret = adv + vals
    B = Tn * n
    buf = dict(feat=torch.stack(feats).reshape(B, -1),
               act=torch.stack(acts).reshape(B, -1),
               logp=torch.stack(logps).reshape(B,),
               adv=adv.reshape(B,),
               ret=ret.reshape(B,))
    # advantage normalization
    adv_b = buf["adv"]
    buf["adv"] = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    return buf, ret_ep.mean().item()


def exp3_fault_adaptation(seeds=range(6), n=256, device=DEV, log=print,
                          ppo_iters=12, envs_per_iter=1, task="push",
                          enc_epochs=None, enc_steps=None):
    """Mid-episode actuator fault; the reviewer-proof condition matrix.

    Deployment fault: t=75, gain x0.5, tau x2 (unsignaled, fixed schedule).
    Training uses per-episode RANDOM faults (t ~ U{40..110}, mults ~ U) so
    domain randomization cannot memorize the deployment timing.

    Conditions (all policies are PPO ActorCritic, frozen after training):
      blind   : obs only, trained on healthy env
      dr      : obs only, trained WITH randomized mid-episode faults
                (strongest inference-free baseline)
      zcond   : [obs, z_est(t)], online GRU posterior, healthy training
      zfault  : [obs, z_est(t)], online posterior, fault training
      rma     : RMA protocol (Kumar et al. 2021) re-implemented in-suite:
                stage 1 = oracle-conditioned teacher trained WITH randomized
                faults (its own t=75 number is the RMA stage-1 reference);
                stage 2 = GRU student distills the teacher's NORMALIZED
                privileged vector from on-policy (o_t, a_{t-1}) history
      oracle  : [obs, z_true(t)] (upper bound; also the RMA teacher)

    Paired stats: zcond-blind, dr-blind, rma-dr, zfault-dr, oracle-rma,
    zcond-rma (the two claims that matter: inference beats randomization,
    and distillation of the privileged vector is NOT enough - the response
    features the teacher's vector captured are what count).
    """
    log = log or (lambda *a, **k: None)
    n_cond = 6
    per_seed = []
    for s in seeds:
        set_seed(7 + s)
        env = PushWorld(n=n, device=device, seed=7 + s, task=task)
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        _train_encoder(env, enc, device, seed=s,
                       epochs=(enc_epochs if enc_epochs is not None else
                               (2 if n <= 64 else 6)),
                       steps_per=(enc_steps if enc_steps is not None else
                                  (10 if n <= 64 else 40)),
                       n_cycles=3)

        def make_ac(din):
            return ActorCritic(din, ACT_DIM).to(device)

        # ---- training loop shared by all conditions ---------------------
        def train_policy(name, din, ac, enc_or_student, with_rand_faults):
            for it in range(ppo_iters):
                ep_rets = []
                for rep in range(envs_per_iter):
                    e = PushWorld(n=n, device=device,
                                  seed=9000 + s * 100 + it * 10 + rep,
                                  fault_at=(75 if with_rand_faults else None),
                                  rand_fault=with_rand_faults, task=task)
                    e.reset()
                    buf, ep_ret = _run_ppo_env3(
                        e, ac, enc_or_student, device, name, task)
                    ppo_update(ac, buf)
                    ep_rets.append(ep_ret)
                if it % 4 == 0 or it == ppo_iters - 1:
                    log(f"      {name} it {it}: return "
                        f"{sum(ep_rets) / len(ep_rets):.2f}")
            return ac

        acs = {}
        acs["blind"] = train_policy("blind", OBS_DIM, make_ac(OBS_DIM),
                                    None, False)
        log(f"    E3 seed {s} trained blind")
        acs["dr"] = train_policy("dr", OBS_DIM, make_ac(OBS_DIM),
                                 None, True)
        log(f"    E3 seed {s} trained dr")
        acs["zcond"] = train_policy("zcond", OBS_DIM + PROP_DIM,
                                    make_ac(OBS_DIM + PROP_DIM), enc, False)
        log(f"    E3 seed {s} trained zcond")
        acs["zfault"] = train_policy("zfault", OBS_DIM + PROP_DIM,
                                     make_ac(OBS_DIM + PROP_DIM), enc, True)
        log(f"    E3 seed {s} trained zfault")

        # RMA stage 1: oracle teacher WITH randomized faults
        acs["oracle"] = train_policy("oracle", OBS_DIM + PROP_DIM,
                                     make_ac(OBS_DIM + PROP_DIM), None, True)
        log(f"    E3 seed {s} trained oracle (RMA stage-1 teacher)")

        # RMA stage 2: distill the teacher's normalized privileged vector
        student = RMAStudent(OBS_DIM, ACT_DIM).to(device)
        _train_rma_student(acs["oracle"], student, s, n, device, task)
        log(f"    E3 seed {s} distilled RMA student")

        # ---- deployment fault eval --------------------------------------
        def run_fault(ac, zmode, zsrc, seed):
            e = PushWorld(n=n, device=device, seed=seed, fault_at=75,
                          task=task)
            e.reset()
            hist_o, hist_a = None, None
            rets = []
            obs = e.obs()
            a = None
            for t in range(150):
                if zmode is None:
                    feat = obs
                else:
                    if hist_o is None:
                        hist_o = obs[None].repeat(8, 1, 1)
                        hist_a = torch.zeros(8, n, ACT_DIM, device=device)
                    else:
                        hist_o = torch.cat([hist_o[1:], obs[None]])
                        hist_a = torch.cat([hist_a[1:], a[None]])
                    with torch.no_grad():
                        if zmode == "oracle":
                            zz = e.true_props()
                        elif zmode == "encoder":
                            lo = PROP_LOW.to(device)
                            hi = PROP_HIGH.to(device)
                            scale = (hi - lo) / PROP_SCALE.to(device)
                            zn = zsrc(hist_o, hist_a)[-1]
                            zz = lo + (zn + 1) / 2 * scale
                        else:  # student outputs normalized-space already
                            lo = PROP_LOW.to(device)
                            hi = PROP_HIGH.to(device)
                            zn = zsrc(hist_o, hist_a)[-1]
                            zz = lo + (zn + 1) / 2 * PROP_SCALE.to(device)
                    feat = torch.cat([obs, zz / PROP_SCALE.to(device)], -1)
                with torch.no_grad():
                    a, _ = ac.act_value(feat)
                obs, r, done = e.act(a)
                rets.append(r)
                if bool(done.all()):
                    break
            rets = torch.stack(rets)
            return dict(pre=float(rets[:75].sum(0).mean()),
                        post=float(rets[75:].sum(0).mean()))

        res = {}
        res["blind"] = run_fault(acs["blind"], None, None, 777 + s)
        res["dr"] = run_fault(acs["dr"], None, None, 777 + s)
        res["zcond"] = run_fault(acs["zcond"], "encoder", enc, 777 + s)
        res["zfault"] = run_fault(acs["zfault"], "encoder", enc, 777 + s)
        res["rma"] = run_fault(acs["oracle"], "student", student, 777 + s)
        res["oracle"] = run_fault(acs["oracle"], "oracle", None, 777 + s)
        per_seed.append(dict(seed=s, **res))
        log(f"  E3 seed {s}: post " +
            " ".join(f"{k} {v['post']:+.3f}" for k, v in res.items()))

    def _post(k):
        return [p[k]["post"] for p in per_seed]

    stats = {
        "zcond_vs_blind": paired_report(_post("zcond"), _post("blind")),
        "dr_vs_blind": paired_report(_post("dr"), _post("blind")),
        "rma_vs_dr": paired_report(_post("rma"), _post("dr")),
        "zfault_vs_dr": paired_report(_post("zfault"), _post("dr")),
        "oracle_vs_rma": paired_report(_post("oracle"), _post("rma")),
        "zcond_vs_rma": paired_report(_post("zcond"), _post("rma")),
        "oracle_vs_zcond": paired_report(_post("oracle"), _post("zcond")),
    }
    return dict(name="fault_adaptation", n_seeds=len(per_seed),
                per_seed=per_seed, stats=stats)


# ------------------------------------------------------------------- E5/E6
def exp5_hold_task(seeds=range(4), n=256, device=DEV, log=print,
                   ppo_iters=20, envs_per_iter=2):
    """Second task family: hold the box at the goal against a constant drift.
    Same 5 properties, same protocol, different dynamics and reward - tests
    whether the property-conditioning result is task-general.
    Headline comparison: zcond vs dr (the two ways to handle hidden params
    without inference); blind = lower bar, oracle = upper bar."""
    log = log or (lambda *a, **k: None)
    # E5 reuses E3's machinery with task="hold" and a 4-condition matrix
    # (rma stage-2 is skipped here to keep budgets comparable; the mechanism
    # question - inference vs randomization - is blind/dr/zcond/oracle)
    per_seed = []
    for s in seeds:
        set_seed(500 + s)
        env = PushWorld(n=n, device=device, seed=500 + s, task="hold")
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        _train_encoder(env, enc, device, seed=s,
                       epochs=(2 if n <= 64 else 6),
                       steps_per=(10 if n <= 64 else 40), n_cycles=3)

        acs = {}
        for name, din, zsrc, faults in (
                ("blind", OBS_DIM, None, False),
                ("dr", OBS_DIM, None, True),
                ("zcond", OBS_DIM + PROP_DIM, enc, False),
                ("oracle", OBS_DIM + PROP_DIM, "oracle", True)):
            ac = ActorCritic(din, ACT_DIM).to(device)
            for it in range(ppo_iters):
                for rep in range(envs_per_iter):
                    e = PushWorld(n=n, device=device,
                                  seed=9700 + s * 100 + it * 10 + rep,
                                  fault_at=(75 if faults else None),
                                  rand_fault=faults, task="hold")
                    e.reset()
                    buf, _ = _run_ppo_env3(e, ac, zsrc, device, name, "hold")
                    ppo_update(ac, buf)
            acs[name] = ac
            log(f"    E5 seed {s} trained {name}")

        def run_hold(ac, zsrc, seed):
            e = PushWorld(n=n, device=device, seed=seed, fault_at=75,
                          task="hold")
            e.reset()
            hist_o, hist_a = None, None
            ret = torch.zeros(n, device=device)
            obs = e.obs()
            a = None
            for t in range(150):
                if zsrc is None:
                    feat = obs
                else:
                    if hist_o is None:
                        hist_o = obs[None].repeat(8, 1, 1)
                        hist_a = torch.zeros(8, n, ACT_DIM, device=device)
                    else:
                        hist_o = torch.cat([hist_o[1:], obs[None]])
                        hist_a = torch.cat([hist_a[1:], a[None]])
                    with torch.no_grad():
                        if zsrc == "oracle":
                            zz = e.true_props()
                        else:
                            lo = PROP_LOW.to(device)
                            hi = PROP_HIGH.to(device)
                            scale = (hi - lo) / PROP_SCALE.to(device)
                            zn = zsrc(hist_o, hist_a)[-1]
                            zz = lo + (zn + 1) / 2 * scale
                    feat = torch.cat([obs, zz / PROP_SCALE.to(device)], -1)
                with torch.no_grad():
                    a, _ = ac.act_value(feat)
                obs, r, done = e.act(a)
                ret = ret + r
                if bool(done.all()):
                    break
            return float(ret.mean().item())

        res = dict(seed=s,
                   blind=run_hold(acs["blind"], None, 690 + s),
                   dr=run_hold(acs["dr"], None, 690 + s),
                   zcond=run_hold(acs["zcond"], enc, 690 + s),
                   oracle=run_hold(acs["oracle"], "oracle", 690 + s))
        per_seed.append(res)
        log(f"  E5 seed {s}: blind {res['blind']:+.2f} dr {res['dr']:+.2f} "
            f"zcond {res['zcond']:+.2f} oracle {res['oracle']:+.2f}")

    def _col(k):
        return [p[k] for p in per_seed]

    stats = {
        "zcond_vs_dr": paired_report(_col("zcond"), _col("dr")),
        "zcond_vs_blind": paired_report(_col("zcond"), _col("blind")),
        "dr_vs_blind": paired_report(_col("dr"), _col("blind")),
        "oracle_vs_zcond": paired_report(_col("oracle"), _col("zcond")),
    }
    return dict(name="hold_task", n_seeds=len(per_seed), per_seed=per_seed,
                stats=stats)


def exp6_identifiability(seeds=range(5), n=256, device=DEV, log=print,
                         epochs=6, steps_per=40):
    """Identifiability kernel (theory probe E6).

    Proposition (box-only contact, gravity-free, torque-free planar dynamics):
      the transition depends on the property vector p = (m, mu, stiff, gain,
      tau) only through m-relative ratios (stiff/m, mu/m) and (gain, tau);
      the absolute scale m is only weakly identifiable (through the ee's
      reaction profile and the box priors).

    Two falsifiable checks, ONE encoder (cheap):
      1. Per-channel audit: the encoder must beat the mean-predictor on the
         identifiable channels (mu, stiff, gain, tau) and may TIE on mass.
         If it ties everywhere, nothing was identified; if it beats the mean
         on mass too, the gain came from the property BOX (priors), which
         the paper must then report honestly as prior-assisted.
      2. Excitation sweep: NRMSE must fall as probe amplitude (contact
         excitation) grows - the practical signature of persistent
         excitation. A flat curve means the encoder is not reading the
         dynamics response at all.
    """
    log = log or (lambda *a, **k: None)
    out_per_seed = []
    for s in seeds:
        set_seed(800 + s)
        env = PushWorld(n=n, device=device, seed=800 + s)
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        _train_encoder(env, enc, device, seed=s, epochs=epochs,
                       steps_per=steps_per, n_cycles=3)
        lo = PROP_LOW.to(device)
        hi = PROP_HIGH.to(device)
        span = hi - lo
        scale = span / PROP_SCALE.to(device)

        # ---- check 1: per-channel audit on held-out fresh episodes ------
        g2 = torch.Generator(device="cpu").manual_seed(s + 909)
        env2 = PushWorld(n=n, device=device, seed=8100 + s)
        from .wm import collect_dataset
        data2 = collect_dataset(env2, n_trajs=4, T=150,
                                policy=(lambda o: _probe_policy(o, g2)),
                                device=device)
        obs2, act2, props2 = (data2["obs"].cpu(), data2["act"].cpu(),
                              data2["props"].cpu())
        L = 32
        g = torch.Generator(device="cpu").manual_seed(s)
        ts = torch.randint(L, obs2.shape[0], (2048,), generator=g)
        idx = torch.randint(0, n, (2048,), generator=g)
        ho = torch.stack([obs2[ts - L + 1 + j, idx] for j in range(L)]).to(device)
        ha = torch.stack([act2[ts - L + 1 + j, idx] for j in range(L)]).to(device)
        z = props2[ts, idx].to(device)
        with torch.no_grad():
            zn = enc(ho, ha)[-1]
        pred = lo + (zn + 1) / 2 * scale
        per_ch = ((pred - z).square().mean(0).sqrt() / span)   # [5]
        mean_pred = z.mean(0)
        per_ch_mean = ((mean_pred - z).square().mean(0).sqrt() / span)
        # channel order: mass, mu, stiff, gain, tau
        ident = ["mu", "stiff", "gain", "tau"]
        ident_idx = [1, 2, 3, 4]
        beats_ident = all(per_ch[i] < per_ch_mean[i] for i in ident_idx)
        mass_gain = float(per_ch_mean[0] - per_ch[0])
        # ---- check 2: excitation amplitude sweep ------------------------
        sweep = []
        for amp in (0.2, 0.4, 0.6, 0.8):
            gs = torch.Generator(device="cpu").manual_seed(s + 700 + int(amp * 10))
            env3 = PushWorld(n=n, device=device, seed=8200 + s + int(amp * 10))
            d3 = collect_dataset(env3, n_trajs=3, T=150,
                                 policy=(lambda o, _a=amp: _probe_policy(
                                     o, gs, amp=_a, noise=0.1)),
                                 device=device)
            o3, a3, p3 = d3["obs"].cpu(), d3["act"].cpu(), d3["props"].cpu()
            ts3 = torch.randint(L, o3.shape[0], (1024,), generator=g)
            idx3 = torch.randint(0, n, (1024,), generator=g)
            ho3 = torch.stack([o3[ts3 - L + 1 + j, idx3] for j in range(L)]).to(device)
            ha3 = torch.stack([a3[ts3 - L + 1 + j, idx3] for j in range(L)]).to(device)
            z3 = p3[ts3, idx3].to(device)
            with torch.no_grad():
                zn3 = enc(ho3, ha3)[-1]
            pred3 = lo + (zn3 + 1) / 2 * scale
            # identifiable-channel NRMSE only (mass is confounded by priors)
            ch = ((pred3 - z3).square().mean(0).sqrt() / span)[ident_idx].mean()
            sweep.append(float(ch))
        out_per_seed.append(dict(seed=s,
                                 per_ch=[float(x) for x in per_ch],
                                 per_ch_mean=[float(x) for x in per_ch_mean],
                                 beats_ident=beats_ident,
                                 mass_prior_gain=mass_gain,
                                 sweep=sweep))
        log(f"  E6 seed {s}: ident beats mean: {beats_ident}, "
            f"mass gain {mass_gain:+.4f}, sweep " +
            ",".join(f"{v:.3f}" for v in sweep))

    import numpy as _np
    n_ident = int(sum(p["beats_ident"] for p in out_per_seed))
    sweep_means = _np.array([p["sweep"] for p in out_per_seed]).mean(0)
    monotone = bool(_np.all(_np.diff(sweep_means) < 0))
    stats = {
        "ident_channels_beat_mean": f"{n_ident}/{len(out_per_seed)} seeds",
        "sweep_monotone_decreasing": monotone,
        "sweep_mean": [float(v) for v in sweep_means],
        "mu_vs_mean": paired_report([p["per_ch"][1] for p in out_per_seed],
                                    [p["per_ch_mean"][1] for p in out_per_seed]),
        "tau_vs_mean": paired_report([p["per_ch"][4] for p in out_per_seed],
                                     [p["per_ch_mean"][4] for p in out_per_seed]),
        "mass_vs_mean": paired_report([p["per_ch"][0] for p in out_per_seed],
                                      [p["per_ch_mean"][0] for p in out_per_seed]),
    }
    return dict(name="identifiability", n_seeds=len(out_per_seed),
                per_seed=out_per_seed, stats=stats)


def _run_ppo_env3(env, ac, zsrc, device, name, task="push", T=150,
                  gamma=0.99, lam_gae=0.95):
    """PPO rollout for E3's condition matrix.
    zsrc: None (blind/dr) | encoder | oracle. name selects z handling.
    Returns (buf, mean episode return)."""
    from .envs import PROP_LOW as _LO, PROP_HIGH as _HI
    n = env.n
    obs = env.obs()
    hist_o, hist_a = None, None
    feats, acts, logps, vals, rews, dones = [], [], [], [], [], []
    ret_ep = torch.zeros(n, device=device)
    a_prev = None
    zmode = None if name in ("blind", "dr") else (
        "oracle" if name == "oracle" else "encoder")
    for t in range(T):
        if zmode is not None:
            if hist_o is None:
                hist_o = obs[None].repeat(8, 1, 1)
                hist_a = torch.zeros(8, n, ACT_DIM, device=device)
            else:
                hist_o = torch.cat([hist_o[1:], obs[None]])
                hist_a = torch.cat([hist_a[1:], a_prev[None]])
            with torch.no_grad():
                if zmode == "oracle":
                    zz = env.true_props()
                else:
                    lo = PROP_LOW.to(device)
                    hi = PROP_HIGH.to(device)
                    scale = (hi - lo) / PROP_SCALE.to(device)
                    zn = zsrc(hist_o, hist_a)[-1]
                    zz = lo + (zn + 1) / 2 * scale
            feat = torch.cat([obs, zz / PROP_SCALE.to(device)], -1)
        else:
            feat = obs
        with torch.no_grad():
            a, val_t = ac.act_value(feat)
            logp_t = ac.dist(feat).log_prob(a).sum(-1)
        feats.append(feat)
        acts.append(a)
        logps.append(logp_t)
        vals.append(val_t)
        a_prev = a
        obs, r, done = env.act(a)
        rews.append(r)
        dones.append(done)
        ret_ep += r
        if bool(done.all()):
            break
    Tn = len(rews)
    rews = torch.stack(rews)
    dones = torch.stack(dones)
    vals = torch.stack(vals)
    with torch.no_grad():
        adv = torch.zeros_like(rews)
        lastgae = torch.zeros(n, device=device)
        for t in reversed(range(Tn)):
            nextv = vals[t + 1] if t + 1 < Tn else torch.zeros(n, device=device)
            mask = (~dones[t]).float()
            delta = rews[t] + gamma * nextv * mask - vals[t]
            lastgae = delta + gamma * lam_gae * mask * lastgae
            adv[t] = lastgae
        ret = adv + vals
    B = Tn * n
    buf = dict(feat=torch.stack(feats).reshape(B, -1),
               act=torch.stack(acts).reshape(B, -1),
               logp=torch.stack(logps).reshape(B,),
               adv=adv.reshape(B,),
               ret=ret.reshape(B,))
    adv_b = buf["adv"]
    buf["adv"] = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    return buf, ret_ep.mean().item()


def _train_rma_student(teacher, student, seed, n, device, task="push",
                       iters=6, lr=1e-3):
    """RMA stage-2 distillation: roll the FROZEN oracle-conditioned teacher in
    the training env WITH randomized faults, and regress the student's GRU
    output (from (o_t, a_{t-1}) history) onto the teacher's normalized
    privileged input vector. On-policy data is what makes the student learn
    the *behaviorally sufficient* projection, not just the properties.
    """
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    for it in range(iters):
        e = PushWorld(n=n, device=device, seed=6600 + seed * 10 + it,
                      fault_at=75, rand_fault=True, task=task)
        e.reset()
        hist_o, hist_a = None, None
        loss_sum, cnt = 0.0, 0
        obs = e.obs()
        a_prev = torch.zeros(n, ACT_DIM, device=device)
        for t in range(150):
            if hist_o is None:
                hist_o = obs[None].repeat(8, 1, 1)
                hist_a = torch.zeros(8, n, ACT_DIM, device=device)
            else:
                hist_o = torch.cat([hist_o[1:], obs[None]])
                hist_a = torch.cat([hist_a[1:], a_prev[None]])
            with torch.no_grad():
                zt = e.true_props()
                feat = torch.cat([obs, zt / PROP_SCALE.to(device)], -1)
                a, _ = teacher.act_value(feat)
            zn = student(hist_o, hist_a)[-1]              # [n, prop], normalized
            tgt = (zt - PROP_LOW.to(device)) / (PROP_HIGH.to(device)
                                                - PROP_LOW.to(device)) * 2 - 1
            loss = (zn - tgt).square().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            cnt += 1
            a_prev = a
            obs, r, done = e.act(a)
            if bool(done.all()):
                break
    return loss_sum / max(1, cnt)


# ------------------------------------------------------------------- E4
def exp4_cross_modal(seeds=range(5), n=256, device=DEV, log=print, ppo_iters=12,
                     envs_per_iter=1):
    """Narrow-box (central 50%) control training; eval on central/edge/2-edge strata.
    blind vs zcond (online posterior)."""
    log = log or (lambda *a, **k: None)
    per_seed = []
    for s in seeds:
        set_seed(11 + s)
        lo, hi = PROP_LOW.to(device), PROP_HIGH.to(device)
        mid = 0.5 * (lo + hi)
        narrow = (mid - 0.25 * (hi - lo), mid + 0.25 * (hi - lo))
        env = PushWorld(n=n, device=device, seed=11 + s, prop_box=narrow)
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        _train_encoder(env, enc, device, seed=s,
                       epochs=(2 if n <= 64 else 6), steps_per=(10 if n <= 64 else 40))

        def train_policy(name, din):
            ac = ActorCritic(din, ACT_DIM).to(device)
            for it in range(ppo_iters):
                ep_rets = []
                for rep in range(envs_per_iter):
                    e = PushWorld(n=n, device=device,
                                  seed=9500 + s * 100 + it * 10 + rep,
                                  prop_box=narrow)
                    e.reset()
                    buf, _ = _run_ppo_env(e, ac, enc, device,
                                          zcond=(name != "blind"), oracle=False)
                    ppo_update(ac, buf)
                    ep_rets.append(0.0)
                if it % 4 == 0 or it == ppo_iters - 1:
                    log(f"      {name} it {it}: trained")
            return ac

        acs = {}
        for name, din in (("blind", OBS_DIM), ("zcond", OBS_DIM + PROP_DIM)):
            acs[name] = train_policy(name, din)
            log(f"    E4 seed {s} trained {name}")

        def strat_eval(ac, name, seed):
            outs = []
            for stratum in range(3):
                g2 = torch.Generator(device="cpu").manual_seed(seed * 31 + stratum)
                base = mid + (torch.rand((n, PROP_DIM), generator=g2).to(device)
                              - 0.5) * 0.2 * (hi - lo)
                props_s = base.clone()
                ar = torch.arange(n, device=device)
                if stratum == 1:
                    kcol = torch.randint(0, PROP_DIM, (n,), generator=g2).to(device)
                    edge = torch.where(torch.rand((n,), generator=g2).to(device) < 0.5,
                                       lo[kcol], hi[kcol])
                    props_s[ar, kcol] = edge
                elif stratum == 2:
                    k1 = torch.randint(0, PROP_DIM, (n,), generator=g2).to(device)
                    k2 = (k1 + 1 + torch.randint(0, PROP_DIM - 1, (n,),
                                                 generator=g2).to(device)) % PROP_DIM
                    c1 = torch.rand((n,), generator=g2).to(device) < 0.5
                    c2 = torch.rand((n,), generator=g2).to(device) < 0.5
                    props_s[ar, k1] = torch.where(c1, lo[k1], hi[k1])
                    props_s[ar, k2] = torch.where(c2, lo[k2], hi[k2])
                e = PushWorld(n=n, device=device, seed=seed * 31 + stratum)
                e.props = props_s.clone()
                e.reset()
                e.props = props_s.clone()
                hist_o, hist_a = None, None
                ret = torch.zeros(n, device=device)
                obs = e.obs()
                a = None
                for t in range(150):
                    if name == "blind":
                        feat = obs
                    else:
                        if hist_o is None:
                            hist_o = obs[None].repeat(8, 1, 1)
                            hist_a = torch.zeros(8, n, ACT_DIM, device=device)
                        else:
                            hist_o = torch.cat([hist_o[1:], obs[None]])
                            hist_a = torch.cat([hist_a[1:], a[None]])
                        with torch.no_grad():
                            lo_ = PROP_LOW.to(device); hi_ = PROP_HIGH.to(device)
                            scale_ = (hi_ - lo_) / PROP_SCALE.to(device)
                            zn = enc(hist_o, hist_a)[-1]
                            zz = lo_ + (zn + 1) / 2 * scale_
                        feat = torch.cat([obs, zz / PROP_SCALE.to(device)], -1)
                    with torch.no_grad():
                        a, _ = ac.act_value(feat)
                    obs, r, done = e.act(a)
                    ret += r
                    if bool(done.all()):
                        break
                outs.append(float(ret.mean().item()))
            return outs

        res = dict(seed=s,
                   blind=strat_eval(acs["blind"], "blind", 888 + s),
                   zcond=strat_eval(acs["zcond"], "zcond", 888 + s))
        per_seed.append(res)
        log(f"  E4 seed {s}: central {res['blind'][0]:+.3f}/{res['zcond'][0]:+.3f} "
            f"edge {res['blind'][1]:+.3f}/{res['zcond'][1]:+.3f}")
    stats = {
        "zcond_vs_blind_edge": paired_report([p["zcond"][1] for p in per_seed],
                                             [p["blind"][1] for p in per_seed]),
        "zcond_vs_blind_double": paired_report([p["zcond"][2] for p in per_seed],
                                               [p["blind"][2] for p in per_seed]),
    }
    return dict(name="cross_modal", n_seeds=len(per_seed), per_seed=per_seed,
                stats=stats)
