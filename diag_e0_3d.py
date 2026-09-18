"""Diagnose 3-D property identifiability: train loss trajectory + held-out eval.

Phase 1 (train): python diag_e0_3d.py train
Phase 2 (eval):  python diag_e0_3d.py eval
"""
import sys
import time

import numpy as np
import torch

torch.set_num_threads(6)

from pec2.envs import PushWorld, PROP_DIM, OBS_DIM, ACT_DIM
from pec2.policy import PropertyEncoder
from pec2.experiments import _train_encoder, _probe_policy, collect_dataset

device = "cpu"


def train_phase():
    env = PushWorld(n=96, device=device, seed=1)
    enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
    t0 = time.time()
    # monkeypatch the training loop? No - run it in chunks by calling with
    # more epochs but capture loss trajectory via a wrapped optimizer is
    # overkill; instead run 5 epochs at a time and re-eval each chunk.
    losses = []
    evals = []
    for chunk in range(5):
        info = _train_encoder(env, enc, device, seed=chunk, epochs=5,
                              n_trajs=8)
        losses.append(info["final_loss"])
        nrmse, nrmse_mean = eval_phase(enc, info)
        evals.append((nrmse, nrmse_mean))
        print(f"chunk {chunk}: loss {info['final_loss']:.4f} "
              f"nrmse {nrmse:.4f} vs mean {nrmse_mean:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    torch.save({"state": enc.state_dict()}, "diag_enc.pt")
    print("LOSS_TRAJ", losses, flush=True)
    print("EVAL_TRAJ", evals, flush=True)


def eval_phase(enc=None, info=None):
    if enc is None:
        enc = PropertyEncoder(OBS_DIM, ACT_DIM).to(device)
        enc.load_state_dict(torch.load("diag_enc.pt")["state"])
        info = {"lo": torch.tensor([0.0]), "span": torch.tensor([1.0])}
    from pec2.envs import PROP_LOW, PROP_HIGH
    lo, span = PROP_LOW.clone(), (PROP_HIGH - PROP_LOW)
    env2 = PushWorld(n=96, device=device, seed=2)
    g2 = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        d2 = collect_dataset(env2, n_trajs=6, T=150,
                             policy=lambda o: _probe_policy(o, g2),
                             device=device)
    obs2, act2, props2 = d2["obs"], d2["act"], d2["props"]
    L = 32
    g = torch.Generator(device="cpu").manual_seed(3)
    ts = torch.randint(L, obs2.shape[0], (4096,), generator=g)
    idx = torch.randint(0, 96, (4096,), generator=g)
    ho = torch.stack([obs2[ts - L + 1 + j, idx] for j in range(L)])
    ha = torch.stack([act2[ts - L + 1 + j, idx] for j in range(L)])
    z = props2[ts, idx]
    with torch.no_grad():
        zn = enc(ho, ha)[-1]
    pred = lo + (zn + 1) / 2 * span
    rmse = (pred - z).square().mean(0).sqrt()
    mean_pred = props2.reshape(-1, PROP_DIM).mean(0)
    rmse_mean = (mean_pred - z).square().mean(0).sqrt()
    nrmse = float((rmse / span).mean())
    nrmse_mean = float((rmse_mean / span).mean())
    per = [round(float(x), 3) for x in (rmse / span)]
    if info is not None:
        return nrmse, nrmse_mean
    print("per-prop NRMSE [mass, mu, act_gain, tau, stiff]:", per, flush=True)
    print("MEAN NRMSE", round(nrmse, 4), "MEAN-PRED", round(nrmse_mean, 4),
          flush=True)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "train"
    if phase == "train":
        train_phase()
    else:
        eval_phase()
