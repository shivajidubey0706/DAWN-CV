"""Configuration: YAML files merged as base -> variant -> experiment overrides.

configs/base.yaml        defaults for every run (hyperparameters, paths)
configs/variants.yaml    the baselines / SG-ERUP variants (B0..B5, E2-x)
configs/<exp>.yaml       experiment files (smoke.yaml, paper_full.yaml, ...)
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
EXPERIMENT_KEYS = ("ablation_variants", "bootstrap", "description")


@dataclass
class FrontEndConfig:
    enabled: bool = True             # False -> no image-processing stage (B0)
    use_bpw: bool = True
    use_kbl: bool = True
    gate_mode: str = "one"           # one | zero | learned | global | fixed
    gate_grid: int | None = None     # None -> encoder feature-map size (8x8 at 256 input)
    gate_kbl: bool = False           # E2-g: gate the KBL output as well
    b0: float = 2.0                  # gate bias init (P4)
    use_gate_features: bool = True   # E2-f
    use_gate_cues: bool = True       # E2-f
    n_intervals: int = 8             # BPW L (unspecified in [8]; fixed and reported)
    encoder_input: int = 256         # assumption (paper Sec. V-C)
    kbl_kernel_scale: float = 1.0
    cue_radius: int = 7              # r
    tau_hi: float = 0.95             # near-saturation threshold
    tau_b: float = 0.6               # bright-region weight centre
    s: float = 0.05                  # bright-region weight temperature


@dataclass
class LossConfig:
    lambda_s: float = 0.0            # saturation-aware penalty weight
    lambda_tv: float = 0.0           # gate smoothness weight
    eta: float = 1.0                 # second-term weight in L_sat
    sat_on: str = "blended"          # blended (I~, default) | final (I^, E2-h)


@dataclass
class AugConfig:
    p_bpw: float = 0.5               # probability of a random global BPW curve
    bpw_strength: float = 1.0
    p_sha: float = 0.0               # probability of spatially heterogeneous augmentation
    sha_scatter_prob: float = 0.0    # optional scattering-model branch of SHA
    hflip: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 6
    imgsz: int = 448
    lr0: float = 1e-4
    lr_min: float = 1e-6
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seed: int = 0
    amp: bool = True
    freeze_detector: bool = True
    max_train_images: int | None = None
    max_val_images: int | None = None
    val_fraction: float = 0.1
    num_workers: int = 0
    deterministic: bool = False
    log_every: int = 20


@dataclass
class RunConfig:
    name: str = "run"
    variant: str = "sg_full"
    detector_weights: str = "weights/yolov3u.pt"
    train_dataset: str = "voc"
    protocol_classes: list[str] = field(default_factory=lambda: ["person", "bicycle", "car", "bus", "motorbike"])
    eval_datasets: list[str] = field(default_factory=lambda: ["dawn"])
    seeds: list[int] = field(default_factory=lambda: [0])
    conf_eval: float = 0.001
    conf_pr: float = 0.25
    iou_nms: float = 0.6
    output_dir: str = "runs"
    frontend: FrontEndConfig = field(default_factory=FrontEndConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build(cls, d: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        sub = {"frontend": FrontEndConfig, "loss": LossConfig, "aug": AugConfig, "train": TrainConfig}.get(k)
        kwargs[k] = _build(sub, v) if sub and cls is RunConfig else v
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_variants() -> dict[str, dict]:
    return load_yaml(CONFIG_DIR / "variants.yaml")["variants"]


def variant_names() -> list[str]:
    return list(load_variants())


def make_config(experiment: str | Path | dict | None = None, variant: str | None = None,
                overrides: dict | None = None) -> RunConfig:
    d = load_yaml(CONFIG_DIR / "base.yaml")
    exp = experiment if isinstance(experiment, dict) else (load_yaml(experiment) if experiment else {})
    d = deep_merge(d, exp)
    v = variant or d.get("variant", "sg_full")
    variants = load_variants()
    if v not in variants:
        raise KeyError(f"unknown variant {v!r}; choose from {sorted(variants)}")
    spec = {k: val for k, val in variants[v].items() if k not in ("label", "description", "paper_id")}
    d = deep_merge(d, spec)
    d = deep_merge(d, overrides or {})
    d["variant"] = v
    for k in EXPERIMENT_KEYS:  # read by scripts/run_ablation.py, not part of a single run
        d.pop(k, None)
    return _build(RunConfig, d)


def parse_overrides(items: list[str]) -> dict:
    """Parse CLI overrides like train.epochs=3 frontend.b0=0."""
    out: dict = {}
    for item in items:
        key, _, raw = item.partition("=")
        val = yaml.safe_load(raw)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out
