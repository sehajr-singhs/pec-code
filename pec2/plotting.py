"""Figures for E0-E4 (matplotlib only, no seaborn)."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _errbar(ax, x, y, ci, label, color):
    y, ci = np.asarray(y), np.asarray(ci)
    ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], label=label,
                color=color, marker="o", capsize=3)


def make_figures(res, out_dir):
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    # --- F2: clock sweep --------------------------------------------------
    try:
        dts = res["e2"]["dts"]
        ks = res["e2"]["ks"]
        fig, axes = plt.subplots(1, len(dts), figsize=(3.2 * len(dts), 2.8),
                                 sharey=False)
        if len(dts) == 1:
            axes = [axes]
        for ax, m in zip(axes, dts):
            o = [res["e2"]["mse"]["latent_ode"][str(m)][str(k)] for k in ks]
            t = [res["e2"]["mse"]["discrete_tick"][str(m)][str(k)] for k in ks]
            ax.plot(ks, o, "o-", label="latent-ODE")
            ax.plot(ks, t, "s--", label="fixed-tick")
            ax.set_xlabel("horizon k (steps)")
            ax.set_title(f"clock dt = {m}·Δt")
            ax.set_yscale("log")
        axes[0].set_ylabel("open-loop MSE")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "fig_clock.pdf"))
        fig.savefig(os.path.join(out_dir, "figures", "fig_clock.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("fig_clock skipped:", e)

    # --- F1: gate ----------------------------------------------------------
    try:
        per = res["e1"]["per_seed"]
        modes = ["honest", "noise", "gate"]
        means = [np.mean([p[m] for p in per]) for m in modes]
        cis = [boot_ci_local([p[m] for p in per]) for m in modes]
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        ax.bar(range(3), means, yerr=[[m - c[0] for m, c in zip(means, cis)],
                                      [c[1] - m for m, c in zip(means, cis)]],
               capsize=4, color=["#888", "#5b8", "#d54"])
        ax.set_xticks(range(3)); ax.set_xticklabels(["plain", "noise", "gate"])
        ax.set_ylabel("real-env return")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "fig_gate.pdf"))
        fig.savefig(os.path.join(out_dir, "figures", "fig_gate.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("fig_gate skipped:", e)

    # --- F3: fault adaptation ---------------------------------------------
    try:
        per = res["e3"]["per_seed"]
        modes = ["blind", "zcond", "oracle"]
        means = [np.mean([p[m]["post"] for p in per]) for m in modes]
        cis = [boot_ci_local([p[m]["post"] for p in per]) for m in modes]
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        ax.bar(range(3), means, yerr=[[m - c[0] for m, c in zip(means, cis)],
                                      [c[1] - m for m, c in zip(means, cis)]],
               capsize=4, color=["#888", "#5b8", "#bbb"])
        ax.set_xticks(range(3))
        ax.set_xticklabels(["blind", "posterior", "oracle"])
        ax.set_ylabel("post-fault return")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "fig_fault.pdf"))
        fig.savefig(os.path.join(out_dir, "figures", "fig_fault.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("fig_fault skipped:", e)

    # --- F4: cross-modal strata --------------------------------------------
    try:
        per = res["e4"]["per_seed"]
        strata = [0, 1, 2]
        w = 0.35
        fig, ax = plt.subplots(figsize=(3.8, 2.8))
        for i, mode in enumerate(["blind", "zcond"]):
            means = [np.mean([p[mode][s] for p in per]) for s in strata]
            cis = [boot_ci_local([p[mode][s] for p in per]) for s in strata]
            ax.bar(np.array(strata) + (i - 0.5) * w, means, width=w,
                   label=mode, color=["#888", "#5b8"][i],
                   yerr=[[m - c[0] for m, c in zip(means, cis)],
                         [c[1] - m for m, c in zip(means, cis)]], capsize=3)
        ax.set_xticks(strata)
        ax.set_xticklabels(["central", "one edge", "two edges"])
        ax.set_ylabel("real-env return")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "fig_crossmodal.pdf"))
        fig.savefig(os.path.join(out_dir, "figures", "fig_crossmodal.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print("fig_crossmodal skipped:", e)


def boot_ci_local(x, n=2000, seed=0):
    x = np.asarray(x, float)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
