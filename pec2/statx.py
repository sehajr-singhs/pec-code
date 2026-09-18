"""Statistics: bootstrap CIs, paired tests, effect sizes."""
from __future__ import annotations
import numpy as np
from scipy import stats as sps


def boot_ci(x, n=2000, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    g = np.random.default_rng(seed)
    idx = g.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    return [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))]


def paired_report(a, b):
    """Paired stats for a-b (per-seed pairs)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(diff=float(d.mean()), ci=boot_ci(d))
    if len(d) >= 6:
        try:
            w = sps.wilcoxon(a, b)
            out["wilcoxon_p"] = float(w.pvalue)
        except ValueError:
            pass
        pooled = np.std(np.concatenate([a, b])) + 1e-9
        out["cohens_d"] = float(d.mean() / pooled)
    return out
