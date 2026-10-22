"""Aggregate saved evaluation JSONs into tables / statistics / plots. Reads only measured outputs."""
from __future__ import annotations

import csv
from pathlib import Path

from .config import load_variants
from .stats import holm, mean_ci, paired_comparison
from .utils import read_json

METRICS = [
    ("map50", lambda m: m.get("map50")),
    ("map50_95", lambda m: m.get("map50_95")),
    ("precision", lambda m: m.get("precision")),
    ("recall", lambda m: m.get("recall")),
    ("ap_small", lambda m: (m.get("sizes") or {}).get("small", {}).get("ap")),
    ("ap_medium", lambda m: (m.get("sizes") or {}).get("medium", {}).get("ap")),
    ("ap_large", lambda m: (m.get("sizes") or {}).get("large", {}).get("ap")),
    ("recall_dark", lambda m: ((m.get("brightness_recall") or {}).get("dark") or {}).get("recall")),
    ("recall_medium", lambda m: ((m.get("brightness_recall") or {}).get("medium") or {}).get("recall")),
    ("recall_bright", lambda m: ((m.get("brightness_recall") or {}).get("bright") or {}).get("recall")),
    ("nsr", lambda m: (m.get("mechanism") or {}).get("nsr")),
    ("brb", lambda m: (m.get("mechanism") or {}).get("brb")),
    ("nsr_final", lambda m: (m.get("mechanism") or {}).get("nsr_final")),
    ("brb_final", lambda m: (m.get("mechanism") or {}).get("brb_final")),
]

# H1 family: DAWN-sand, DAWN-fog, RTTS (paper Sec. III-E); only cells that were evaluated enter.
CONFIRMATORY = [("dawn", "sand"), ("dawn", "fog"), ("rtts", "all")]


def load_results(exp_dir: Path) -> list[dict]:
    out = []
    for p in sorted(exp_dir.glob("*.json")):
        if p.name in ("summary.json",):
            continue
        try:
            r = read_json(p)
        except Exception:  # noqa: BLE001
            continue
        if r.get("status") == "evaluated":
            r["_file"] = p.name
            out.append(r)
    return out


def summarize_experiment(exp_dir: Path, reference: str = "b2_erup") -> dict:
    results = load_results(exp_dir)
    labels = {k: v.get("label", k) for k, v in load_variants().items()}
    # cells[(variant, dataset, subset)][seed] = metrics dict
    cells: dict[tuple, dict[int, dict]] = {}
    meta: dict[tuple, dict] = {}
    for r in results:
        subsets = {"all": r["overall"], **r.get("by_weather", {})}
        for sub, m in subsets.items():
            key = (r["variant"], r["dataset"], sub)
            cells.setdefault(key, {})[r["seed"]] = m
            meta[key] = {"n_images": m.get("n_images"), "n_gt": m.get("n_gt"), "trained": r.get("trained"),
                         "settings": r.get("settings", {})}
    rows = []
    for (v, ds, sub), by_seed in sorted(cells.items()):
        row = {"variant": v, "label": labels.get(v, v), "dataset": ds, "subset": sub, "n_seeds": len(by_seed),
               "seeds": sorted(by_seed), **{k: meta[(v, ds, sub)][k] for k in ("n_images", "n_gt", "trained")}}
        for name, get in METRICS:
            row[name] = mean_ci([get(m) for m in by_seed.values()])
        rows.append(row)

    comparisons = []
    for (v, ds, sub), by_seed in sorted(cells.items()):
        ref = cells.get((reference, ds, sub))
        if v == reference or not ref:
            continue
        a = {s: m.get("map50") for s, m in by_seed.items()}
        b = {s: m.get("map50") for s, m in ref.items()}
        c = paired_comparison(a, b)
        c.update(variant=v, reference=reference, dataset=ds, subset=sub, metric="map50")
        comparisons.append(c)
    fam = {f"{c['variant']}|{c['dataset']}|{c['subset']}": c["wilcoxon_p"] for c in comparisons
           if c["variant"] == "sg_full" and (c["dataset"], c["subset"]) in CONFIRMATORY}
    holm_res = holm(fam) if fam else {}

    _write_csv(exp_dir / "summary.csv", rows)
    _write_md(exp_dir / "summary.md", rows, comparisons)
    _plot(exp_dir, rows)
    return {"experiment": exp_dir.name, "n_result_files": len(results), "reference": reference,
            "rows": rows, "comparisons": comparisons, "holm_confirmatory": holm_res,
            "variants_evaluated": sorted({r["variant"] for r in results}),
            "datasets": sorted({r["dataset"] for r in results}),
            "seeds": sorted({r["seed"] for r in results})}


def _fmt(c: dict, scale: float = 100.0) -> str:
    if not c or c.get("mean") is None:
        return "Not evaluated"
    s = f"{c['mean'] * scale:.2f}"
    if c.get("std") is not None:
        s += f" ± {c['std'] * scale:.2f}"
    return s


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "label", "dataset", "subset", "n_seeds", "n_images", "n_gt"]
                   + [f"{n}_{s}" for n, _ in METRICS for s in ("mean", "std")])
        for r in rows:
            w.writerow([r["variant"], r["label"], r["dataset"], r["subset"], r["n_seeds"], r["n_images"], r["n_gt"]]
                       + [x for n, _ in METRICS for x in (r[n]["mean"], r[n]["std"])])


def _write_md(path: Path, rows: list[dict], comps: list[dict]) -> None:
    lines = ["# Experiment summary (generated from result files)", "",
             "Values in % (mean ± s.d. over seeds). 'Not evaluated' = no result file.", "",
             "| Variant | Dataset | Subset | Seeds | Images | mAP@0.5 | mAP@[.5:.95] | P | R | Recall bright | NSR | BRB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['dataset']} | {r['subset']} | {r['n_seeds']} | {r['n_images']} | "
                     f"{_fmt(r['map50'])} | {_fmt(r['map50_95'])} | {_fmt(r['precision'])} | {_fmt(r['recall'])} | "
                     f"{_fmt(r['recall_bright'])} | {_fmt(r['nsr'])} | {_fmt(r['brb'])} |")
    if comps:
        lines += ["", "## Paired differences vs reference (mAP@0.5 points)", "",
                  "| Variant | Dataset | Subset | Pairs | Δ mean | 95% CI | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
        for c in comps:
            d = c["diff"]
            ci = f"[{d['ci'][0] * 100:.2f}, {d['ci'][1] * 100:.2f}]" if d.get("ci") else "n/a"
            mean = f"{d['mean'] * 100:+.2f}" if d.get("mean") is not None else "n/a"
            p = f"{c['wilcoxon_p']:.3f}" if c["wilcoxon_p"] is not None else "n/a"
            lines.append(f"| {c['variant']} | {c['dataset']} | {c['subset']} | {c['n_pairs']} | {mean} | {ci} | {p} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(exp_dir: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    for ds in sorted({r["dataset"] for r in rows}):
        rs = [r for r in rows if r["dataset"] == ds]
        subs = sorted({r["subset"] for r in rs}, key=lambda s: (s != "all", s))
        vs = sorted({r["variant"] for r in rs})
        if not vs:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(subs) * max(1, len(vs)) / 3), 3.8))
        width = 0.8 / len(vs)
        for i, v in enumerate(vs):
            ys, es = [], []
            for s in subs:
                c = next((r["map50"] for r in rs if r["variant"] == v and r["subset"] == s), None)
                ys.append((c or {}).get("mean") or 0)
                es.append((c or {}).get("std") or 0)
            ax.bar(np.arange(len(subs)) + i * width, np.array(ys) * 100, width, yerr=np.array(es) * 100, label=v)
        ax.set_xticks(np.arange(len(subs)) + 0.4 - width / 2)
        ax.set_xticklabels(subs)
        ax.set_ylabel("mAP@0.5 (%)")
        ax.set_title(f"{ds}: measured mAP@0.5 by subset")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=.3)
        fig.tight_layout()
        fig.savefig(exp_dir / f"map50_{ds}.png", dpi=110)
        plt.close(fig)
