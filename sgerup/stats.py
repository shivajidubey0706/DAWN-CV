"""Paired multi-seed statistics (paper Sec. V-E): t-intervals, Wilcoxon signed-rank, Holm."""
from __future__ import annotations

import math

import numpy as np


def mean_ci(xs, conf: float = 0.95) -> dict:
    x = np.asarray([v for v in xs if v is not None], dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci": None}
    if n == 1:
        return {"n": 1, "mean": float(x[0]), "std": None, "ci": None}
    from scipy import stats
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = float(stats.t.ppf(0.5 + conf / 2, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "std": sd, "ci": [m - h, m + h]}


def paired_comparison(a: dict[int, float], b: dict[int, float]) -> dict:
    """a, b: seed -> metric. Difference a - b over seeds present in both."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds if a[s] is not None and b[s] is not None]
    out = {"seeds": seeds, "n_pairs": len(d), "diff": mean_ci(d), "wilcoxon_p": None,
           "note": None}
    if len(d) >= 2 and any(v != 0 for v in d):
        from scipy import stats
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except ValueError as e:
            out["note"] = str(e)
    if len(d) < 6:
        out["note"] = (f"only {len(d)} paired seed(s); the paper's protocol uses 8. With n < 6 a two-sided "
                       "Wilcoxon test cannot reach p < 0.05; treat as inconclusive.")
    return out


def holm(pvals: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(((k, p) for k, p in pvals.items() if p is not None), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    for k, p in pvals.items():
        out.setdefault(k, {"p": None, "p_holm": None, "reject": None})
    return out
