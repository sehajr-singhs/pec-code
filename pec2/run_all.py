"""Run E0-E4 with a CLI profile; saves results JSON and figures.

Usage:
    python -m pec2.run_all --profile smoke --seeds 2
    python -m pec2.run_all --profile full  --seeds 6
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch

from . import experiments as E
from .statx import boot_ci

PROFILES = {
    "ci": dict(
        e0=dict(n=96, enc_epochs=3),
        e1=dict(gens=15, pop=16, T=20, n_dream=16),
        e2=dict(dts=(1, 4), ks=(1, 2)),
        e3=dict(n=64, ppo_iters=4, envs_per_iter=2),
        e4=dict(n=64, ppo_iters=4, envs_per_iter=2),
        e5=dict(n=64, ppo_iters=4, envs_per_iter=2),
        e6=dict(n=96, epochs=3, steps_per=20),
    ),
    "smoke": dict(
        e0=dict(n=128, enc_epochs=2),
        e1=dict(gens=40, pop=32, T=30, n_dream=32),
        e2=dict(dts=(1, 2, 4), ks=(1, 2, 4)),
        e3=dict(n=128, envs_per_iter=2),
        e4=dict(n=128, envs_per_iter=2),
        e5=dict(n=128, envs_per_iter=2),
        e6=dict(n=128, epochs=4, steps_per=24),
    ),
    "full": dict(
        e0=dict(n=256, enc_epochs=12),
        e1=dict(gens=150, pop=64, T=40, n_dream=64),
        e2=dict(dts=(1, 2, 4, 8), ks=(1, 2, 4, 8)),
        e3=dict(n=256, ppo_iters=30, envs_per_iter=4),
        e4=dict(n=256, ppo_iters=30, envs_per_iter=4),
        e5=dict(n=256, ppo_iters=24, envs_per_iter=4),
        e6=dict(n=256, epochs=8, steps_per=40),
    ),
    # seed-farm profile: tractable per-seed budget for parallel CPU workers
    # (~35-60 min per seed at 2 threads); statistical power comes from many
    # seeds, not from one heavy seed
    "farm": dict(
        e0=dict(n=128, enc_epochs=8),
        e1=dict(gens=60, pop=32, T=30, n_dream=32),
        e2=dict(dts=(1, 4), ks=(1, 2, 4)),
        e3=dict(n=96, ppo_iters=12, envs_per_iter=2),
        e4=dict(n=96, ppo_iters=12, envs_per_iter=2),
        e5=dict(n=96, ppo_iters=10, envs_per_iter=2),
        e6=dict(n=128, epochs=6, steps_per=30),
        e7=dict(budgets=[250, 500, 1000, 2000]),
        # hold_v2 follow-up farm: approach-curriculum variant (see PREREGISTRATION.md)
        e5v2=dict(n=96, ppo_iters=10, envs_per_iter=2, task="hold_v2"),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="ci", choices=["ci", "smoke", "full"])
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds; overrides profile defaults")
    ap.add_argument("--out", default="results_pec2")
    ap.add_argument("--only", default=None,
                    help="comma list like e0,e3 - runs only these; merges "
                         "with an existing stats file in --out if present")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    log = print

    n_seeds = args.seeds or {"ci": 1, "smoke": 2, "full": 6}[args.profile]
    p = PROFILES[args.profile]

    t0 = time.time()
    results = {"meta": dict(device=dev, profile=args.profile,
                            n_seeds=n_seeds,
                            torch=torch.__version__,
                            wall_seconds=None)}
    only = args.only.split(",") if args.only else None
    stats_path = os.path.join(args.out, "stats_pec2.json")
    if only and os.path.exists(stats_path):
        with open(stats_path) as f:
            results.update(json.load(f))

    def wanted(tag):
        return only is None or tag in only

    if wanted("e0"):
        log(f"[E0] estimator quality ({n_seeds} seeds)")
        results["e0"] = E.exp0_estimator(seeds=range(n_seeds), device=dev,
                                         **p["e0"])
        log(f"    mean normalized RMSE {results['e0']['mean_nrmse']:.3f}")

    if wanted("e2"):
        log(f"[E2] clock sweep ({n_seeds} seeds)")
        results["e2"] = E.exp2_clock_sweep(seeds=range(max(1, n_seeds // 2)),
                                           device=dev, **p["e2"])
        mo = results["e2"]["mse"]["latent_ode"]
        mk = results["e2"]["mse"]["discrete_tick"]
        log(f"    latent-ODE MSE: {json.dumps(mo)}")
        log(f"    fixed-tick MSE: {json.dumps(mk)}")

    if wanted("e1"):
        log(f"[E1] dream gate ({n_seeds} seeds)")
        results["e1"] = E.exp1_gate_vs_noise(seeds=range(n_seeds), device=dev,
                                             **p["e1"])
        log(f"    gate vs noise: {results['e1']['stats']['gate_vs_noise']}")

    if wanted("e3"):
        log(f"[E3] fault adaptation, 6-condition matrix ({n_seeds} seeds)")
        results["e3"] = E.exp3_fault_adaptation(seeds=range(n_seeds),
                                                device=dev, **p["e3"])
        st = results["e3"]["stats"]
        log(f"    zcond vs blind: {st['zcond_vs_blind']}")
        log(f"    zcond vs rma:   {st['zcond_vs_rma']}")
        log(f"    dr vs blind:    {st['dr_vs_blind']}")

    if wanted("e4"):
        log(f"[E4] cross-modal transfer ({n_seeds} seeds)")
        results["e4"] = E.exp4_cross_modal(seeds=range(n_seeds), device=dev,
                                           **p["e4"])
        log(f"    zcond vs blind (edge): {results['e4']['stats']['zcond_vs_blind_edge']}")

    if wanted("e5"):
        log(f"[E5] hold task, second task family ({n_seeds} seeds)")
        results["e5"] = E.exp5_hold_task(seeds=range(n_seeds), device=dev,
                                         **p["e5"])
        log(f"    zcond vs dr: {results['e5']['stats']['zcond_vs_dr']}")

    if wanted("e6"):
        log(f"[E6] identifiability kernel ({n_seeds} seeds)")
        results["e6"] = E.exp6_identifiability(seeds=range(n_seeds), device=dev,
                                               **p["e6"])
        log(f"    ident channels: {results['e6']['stats']['ident_channels_beat_mean']}, "
            f"sweep monotone: {results['e6']['stats']['sweep_monotone_decreasing']}")

    results["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=1)

    from .plotting import make_figures
    make_figures(results, args.out)
    log(f"done in {results['meta']['wall_seconds']} s -> {args.out}/")


if __name__ == "__main__":
    main()
