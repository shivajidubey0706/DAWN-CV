"""Scan every registered dataset, validate annotations, and check split leakage.

    python scripts/validate_datasets.py                 # all datasets, fast
    python scripts/validate_datasets.py --verify-images # also decode-check every image
    python scripts/validate_datasets.py --only dawn voc
"""
import argparse

import _common  # noqa: F401

from sgerup.config import ROOT
from sgerup.data.datasets import check_disjoint, load_samples, registry, scan_dataset
from sgerup.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--verify-images", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / "results" / "datasets"
    summary = {}
    for name in a.only or list(registry()):
        splits = ["trainval", "test"] if registry()[name]["format"] == "voc" else [None]
        for split in splits:
            key = f"{name}_{split}" if split else name
            rep = scan_dataset(name, a.verify_images, split)
            write_json(out_dir / f"{key}.json", rep)
            summary[key] = {k: rep.get(k) for k in ("status", "n_valid_samples", "n_protocol_boxes",
                                                    "n_missing_annotations", "n_issues", "root")}
            print(f"{key:18s} {rep['status']:9s} images={rep.get('n_valid_samples', 0)} "
                  f"boxes={rep.get('n_protocol_boxes', 0)} issues={rep.get('n_issues', 0)}")
    # Leakage: VOC trainval vs VOC test vs DAWN (test-only datasets)
    named = {}
    if registry()["voc"] and (ROOT / registry()["voc"]["root"]).exists():
        named["voc_trainval"] = load_samples("voc", "trainval")
        named["voc_test"] = load_samples("voc", "test")
    named["dawn"] = load_samples("dawn")
    leak = check_disjoint({k: v for k, v in named.items() if v}, by_content=False)
    summary["_leakage_check"] = leak
    write_json(out_dir / "summary.json", summary)
    print("leakage check:", "PASS (disjoint)" if leak["disjoint"] else f"FAIL {leak['overlaps']}", leak["sizes"])


if __name__ == "__main__":
    main()
