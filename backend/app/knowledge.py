"""Local retrieval over the paper text, project documentation and measured results (BM25)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .settings import ROOT

SOURCES = [
    ("paper", ROOT / "knowledge" / "paper.md"),
    ("faq", ROOT / "knowledge" / "faq.md"),
    ("method", ROOT / "docs" / "METHOD_NOTES.md"),
    ("readme", ROOT / "README.md"),
    ("datasets", ROOT / "docs" / "DATASETS.md"),
    ("models", ROOT / "docs" / "MODELS.md"),
    ("architecture", ROOT / "docs" / "ARCHITECTURE.md"),
    ("testing", ROOT / "docs" / "TESTING_REPORT.md"),
]
_TOKEN = re.compile(r"[a-z0-9_]+(?:[.@][0-9]+)?")
_STOP = set("the a an of and or to in is are for on with by as be it this that from at not we our its was were can which".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    source: str
    title: str
    text: str
    kind: str  # documented | measured


def _chunks_from_markdown(source: str, text: str, size: int = 1100) -> list[Chunk]:
    out, title, buf = [], source, []
    def flush():
        body = "\n".join(buf).strip()
        while body:
            out.append(Chunk(source, title, body[:size], "documented"))
            body = body[size - 150:] if len(body) > size else ""
        buf.clear()
    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            title = line.lstrip("# ").strip() or source
        else:
            buf.append(line)
    flush()
    return out


def _chunks_from_paper(text: str, size: int = 1200) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf, title = [], "", "Paper"
    for p in paras:
        m = re.match(r"^([IVX]+\.\s+[A-Z][A-Z ,\-]+|[A-H]\.\s+[A-Z][^\n]{3,60})$", p)
        if m:
            title = p
        if len(buf) + len(p) > size and buf:
            out.append(Chunk("paper", title, buf.strip(), "documented"))
            buf = ""
        buf += " " + re.sub(r"\s+", " ", p)
    if buf.strip():
        out.append(Chunk("paper", title, buf.strip(), "documented"))
    return out


def results_digest() -> list[Chunk]:
    """Compact, factual digest of measured results, so answers can cite actual numbers."""
    out = []
    exp_root = ROOT / "results" / "experiments"
    for summ in sorted(exp_root.glob("*/summary.json")) if exp_root.exists() else []:
        try:
            s = json.loads(summ.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        lines = [f"Experiment '{s['experiment']}' (measured; seeds {s.get('seeds')}; reference {s.get('reference')}):"]
        for r in s.get("rows", []):
            def f(k, scale=100):
                c = r.get(k) or {}
                if c.get("mean") is None:
                    return "n/a"
                return f"{c['mean'] * scale:.2f}" + (f"±{c['std'] * scale:.2f}" if c.get("std") is not None else "")
            lines.append(f"- {r['variant']} on {r['dataset']}/{r['subset']} (n_img={r['n_images']}, seeds={r['n_seeds']}): "
                         f"mAP50={f('map50')} mAP50-95={f('map50_95')} P={f('precision')} R={f('recall')} "
                         f"R_bright={f('recall_bright')} R_dark={f('recall_dark')} NSR={f('nsr')} BRB={f('brb')}")
        for c in s.get("comparisons", []):
            d = c["diff"]
            if d.get("mean") is not None:
                lines.append(f"- paired diff {c['variant']} − {c['reference']} on {c['dataset']}/{c['subset']}: "
                             f"{d['mean'] * 100:+.2f} mAP50 points over {c['n_pairs']} seed pair(s); "
                             f"p={c['wilcoxon_p']}; {c.get('note') or ''}")
        text = "\n".join(lines)
        for i in range(0, len(text), 1500):
            out.append(Chunk(f"results:{s['experiment']}", "Measured results", text[i:i + 1500], "measured"))
    cost = ROOT / "results" / "cost" / "cost.json"
    if cost.exists():
        c = json.loads(cost.read_text(encoding="utf-8"))
        lines = [f"Cost measurement (measured on {c.get('device')} / {c.get('environment', {}).get('gpu')}, "
                 f"{c.get('imgsz')}x{c.get('imgsz')}, batch 1; {c.get('flops_convention')}):"]
        for r in c.get("rows", []):
            lines.append(f"- {r['variant']}: params front end {r['params_frontend']:,}, total {r['params_total']:,}; "
                         f"GFLOPs total {r['gflops_total']}; median latency fp32 {r['latency_fp32']['median_ms']:.2f} ms")
        out.append(Chunk("results:cost", "Measured cost", "\n".join(lines), "measured"))
    return out


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.reload()

    def reload(self) -> None:
        chunks: list[Chunk] = []
        for name, path in SOURCES:
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks += _chunks_from_paper(text) if name == "paper" else _chunks_from_markdown(name, text)
        chunks += results_digest()
        self.chunks = chunks
        self.docs = [tokenize(c.title + " " + c.text) for c in chunks]
        self.df = Counter(t for d in self.docs for t in set(d))
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))

    def search(self, query: str, k: int = 6) -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        n = len(self.docs)
        scores = []
        for d, c in zip(self.docs, self.chunks):
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / self.avgdl))
            if c.kind == "measured" and any(t in q for t in ("result", "results", "map", "map50", "measured",
                                                                  "performance", "nsr", "brb", "recall", "precision")):
                s *= 1.5
            if s > 0:
                scores.append((s, c))
        return sorted(scores, key=lambda x: -x[0])[:k]
