"""E7: where does the latent-ODE's clock advantage emerge? (budget ablation)

The evidence so far is apparently contradictory:
  - 1-D suite (trained to convergence): latent-ODE wins by an order of
    magnitude at coarse clocks,
  - 2-D full run (limited budget): fixed-tick matches or wins at every clock,
  - 3-D unit test (tiny budget): latent-ODE wins at the coarse clock.

These reconcile ONLY if the advantage is non-monotonic in training budget.
This ablation measures capacity-matched open-loop MSE at the coarse eval
clock as a function of gradient-step budget, for both models, on identical
excitation-probe data. Output: the crossover budget (or its absence - both
are reportable), written to exp7_clock_budget.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pec2.envs import PushWorld, OBS_DIM, ACT_DIM  # noqa: E402
from pec2.policy import TinyPolicy  # noqa: E402
from pec2.wm import (LatentODEWorldModel, DiscreteTickWorldModel,  # noqa: E402
                     collect_dataset, train_wm, eval_wm)
from pec2 import experiments as E  # noqa: E402  (probe policy)


def run(seeds=range(3), budgets=(500, 1000, 2000, 4000), device="cpu",
        log=print, m_eval=4, k_eval=4):
    log = log or (lambda *a, **k: None)
    rows = []
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "..", "exp7_clock_budget.json")
    for s in seeds:
        torch.manual_seed(42 + s)
        np.random.seed(42 + s)
        env = PushWorld(n=128, device=device, seed=42 + s)
        g = torch.Generator(device="cpu").manual_seed(s + 31337)
        data = collect_dataset(env, n_trajs=4, T=150,
                               policy=(lambda o: E._probe_policy(o, g)),
                               device=device)
        for B in budgets:
            wm_o = LatentODEWorldModel().to(device)
            wm_t = DiscreteTickWorldModel().to(device)
            lo = train_wm(wm_o, data, device, epochs=max(1, B // 8),
                          steps_per=8, seed=s)
            lt = train_wm(wm_t, data, device, epochs=max(1, B // 8),
                          steps_per=8, seed=s)
            mse_o = eval_wm(wm_o, data, m_eval, k_eval, device, seed=100 + s)
            mse_t = eval_wm(wm_t, data, m_eval, k_eval, device, seed=100 + s)
            rows.append(dict(seed=s, budget=B, ode_mse=mse_o, tick_mse=mse_t,
                             ode_train_loss=lo, tick_train_loss=lt))
            log(f"E7 s{s} B{B}: ode {mse_o:.4f} vs tick {mse_t:.4f} "
                f"-> {'ODE' if mse_o < mse_t else 'tick'} "
                f"({abs(mse_o - mse_t) / max(mse_o, mse_t) * 100:.0f}%)")
            # incremental save: every row survives a kill
            with open(out_path, "w") as f:
                json.dump(dict(name="clock_budget_emergence_partial",
                               m_eval=m_eval, k_eval=k_eval, rows=rows), f,
                          indent=1)
    # crossover analysis
    budgets = sorted({r["budget"] for r in rows})
    curve = {"ode": [], "tick": []}
    for B in budgets:
        for mdl in ("ode", "tick"):
            vals = [r[f"{mdl}_mse"] for r in rows if r["budget"] == B]
            curve[mdl].append(float(np.mean(vals)))
    winner = ["ode" if o < t else "tick" for o, t in zip(curve["ode"], curve["tick"])]
    crossover = None
    for i in range(1, len(winner)):
        if winner[i] != winner[i - 1]:
            crossover = budgets[i]
            break
    out = dict(name="clock_budget_emergence", m_eval=m_eval, k_eval=k_eval,
               budgets=list(budgets), curve=curve, winner=winner,
               crossover_budget=crossover, rows=rows)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    log(f"E7 winner by budget: {dict(zip(budgets, winner))}, "
        f"crossover at B={crossover}")
    return out


if __name__ == "__main__":
    run()
