"""Aggregate seed-farm per-seed JSONs into paper-grade paired statistics.

Reads results/seedfarm/<exp>/seed_*.json, pools per-seed numbers, and writes
aggregated.json with the same paired-report structure the paper cites.

Usage:
    python aggregate_farm.py            # all experiments found
    python aggregate_farm.py e3,e5
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pec2.statx import paired_report, boot_ci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FARM = os.path.join(HERE, "results", "seedfarm")


def load(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(FARM, exp, "seed_*.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def agg_e0(rows):
    per = [r["per_seed"][0] for r in rows]
    a = np.array([p["nrmse"] for p in per])
    m = np.array([p["nrmse_mean"] for p in per])
    return dict(n_seeds=len(per),
                mean_nrmse=float(a.mean()), ci=boot_ci(a),
                mean_predictor_nrmse=float(m.mean()), ci_mean=boot_ci(m),
                encoder_wins=int((a < m).sum()),
                paired=paired_report(a, m))


def agg_e3(rows):
    per = [r["per_seed"][0] for r in rows]
    conds = ("blind", "dr", "zcond", "zfault", "rma", "oracle")
    post = {c: np.array([p[c]["post"] for p in per]) for c in conds}
    pre = {c: np.array([p[c]["pre"] for p in per]) for c in conds}
    pairs = ("zcond_vs_blind", "dr_vs_blind", "rma_vs_dr", "zfault_vs_dr",
             "oracle_vs_rma", "zcond_vs_rma", "oracle_vs_zcond")
    stats = {}
    for pr in pairs:
        a, b = pr.split("_vs_")
        stats[pr] = paired_report(post[a], post[b])
    return dict(n_seeds=len(per),
                post_mean={c: float(v.mean()) for c, v in post.items()},
                post_ci={c: boot_ci(v) for c, v in post.items()},
                pre_mean={c: float(v.mean()) for c, v in pre.items()},
                stats=stats)


def agg_e5(rows):
    per = [r["per_seed"][0] for r in rows]
    conds = ("blind", "dr", "zcond", "oracle")
    col = {c: np.array([p[c] for p in per]) for c in conds}
    stats = {pr: paired_report(col[pr.split("_vs_")[0]], col[pr.split("_vs_")[2]])
             for pr in ("zcond_vs_dr", "zcond_vs_blind", "dr_vs_blind",
                        "oracle_vs_zcond")}
    return dict(n_seeds=len(per),
                mean={c: float(v.mean()) for c, v in col.items()},
                ci={c: boot_ci(v) for c, v in col.items()},
                stats=stats)


def agg_e6(rows):
    per = [r["per_seed"][0] for r in rows]
    n = len(per)
    sweep = np.array([p["sweep"] for p in per]).mean(0)
    per_ch = np.array([p["per_ch"] for p in per])          # [S, 5]
    per_ch_mean = np.array([p["per_ch_mean"] for p in per])
    names = ("mass", "mu", "stiff", "gain", "tau")
    chan = {nm: paired_report(per_ch[:, i], per_ch_mean[:, i])
            for i, nm in enumerate(names)}
    return dict(n_seeds=n,
                ident_channels_beat_mean=f"{int(sum(p['beats_ident'] for p in per))}/{n} seeds",
                sweep_mean=[float(v) for v in sweep],
                sweep_monotone_decreasing=bool(np.all(np.diff(sweep) < 0)),
                channels=chan)


AGGS = {"e0": agg_e0, "e3": agg_e3, "e5": agg_e5, "e6": agg_e6}


def main():
    exps = sys.argv[1].split(",") if len(sys.argv) > 1 else list(AGGS)
    out = {}
    for e in exps:
        rows = load(e)
        if not rows:
            print(f"{e}: no seed files yet")
            continue
        out[e] = AGGS[e](rows)
        print(f"{e}: {out[e]['n_seeds']} seeds aggregated")
    path = os.path.join(FARM, "aggregated.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")
    # print the headline lines
    if "e3" in out:
        s = out["e3"]["stats"]
        for k in ("zcond_vs_blind", "zcond_vs_dr", "zcond_vs_rma", "dr_vs_blind",
                  "rma_vs_dr"):
            if k in s:
                r = s[k]
                print(f"  E3 {k}: diff {r['diff']:+.4f} "
                      f"p={r.get('wilcoxon_p', '-')} d={r.get('cohens_d', '-')}")
    if "e5" in out:
        r = out["e5"]["stats"]["zcond_vs_dr"]
        print(f"  E5 zcond_vs_dr: diff {r['diff']:+.3f} "
              f"p={r.get('wilcoxon_p', '-')} d={r.get('cohens_d', '-')}")
    if "e0" in out:
        o = out["e0"]
        print(f"  E0: encoder {o['mean_nrmse']:.4f} vs mean-pred "
              f"{o['mean_predictor_nrmse']:.4f} (wins {o['encoder_wins']}/{o['n_seeds']})")


if __name__ == "__main__":
    main()
