"""Process-wide state: lazily loaded detector pipeline, result store, dataset cache, training jobs."""
from __future__ import annotations

import collections
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .settings import ROOT, settings

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ModelState:
    def __init__(self):
        self.pipeline = None
        self.error: str | None = None
        self.loading = False
        self.loaded_at: float | None = None
        self.lock = threading.Lock()        # serialises GPU inference
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self.pipeline is not None:
                return
            self.loading = True
            try:
                from sgerup.detector import DetectorError, YOLODetector
                from sgerup.pipeline import Pipeline
                det = YOLODetector(settings.detector_weights, settings.torch_device)
                self.pipeline = Pipeline(det, settings.imgsz)
                self.error = None
                self.loaded_at = time.time()
            except DetectorError as e:
                self.error = str(e)
            except Exception as e:  # noqa: BLE001
                self.error = f"{e.__class__.__name__}: {e}"
            finally:
                self.loading = False

    def get(self):
        if self.pipeline is None:
            self.load()
        if self.pipeline is None:
            raise RuntimeError(self.error or "detector not loaded")
        return self.pipeline


class ResultStore:
    """Bounded in-memory store for downloadable outputs (no files written for uploads)."""

    def __init__(self, cap: int):
        self.cap = cap
        self.items: collections.OrderedDict[str, dict] = collections.OrderedDict()
        self.lock = threading.Lock()

    def put(self, files: dict[str, tuple[bytes, str]], meta: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.lock:
            self.items[rid] = {"files": files, "meta": meta, "created": time.time()}
            while len(self.items) > self.cap:
                self.items.popitem(last=False)
        return rid

    def get(self, rid: str, name: str) -> tuple[bytes, str] | None:
        with self.lock:
            item = self.items.get(rid)
            return None if item is None else item["files"].get(name)


class JobRunner:
    """Runs whitelisted training configs as subprocesses, one at a time."""
    ALLOWED_CONFIGS = {"smoke": "configs/smoke.yaml", "short_real": "configs/short_real.yaml"}

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def running(self) -> dict | None:
        for j in self.jobs.values():
            if j["proc"].poll() is None:
                return j
        return None

    def start(self, config: str, variant: str, seed: int) -> dict:
        from sgerup.config import variant_names
        if config not in self.ALLOWED_CONFIGS:
            raise ValueError(f"config must be one of {sorted(self.ALLOWED_CONFIGS)}")
        if variant not in variant_names():
            raise ValueError("unknown variant")
        with self.lock:
            if self.running():
                raise RuntimeError("a training job is already running")
            jid = uuid.uuid4().hex[:8]
            log_path = ROOT / "logs" / f"job_{jid}.log"
            log_path.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", self.ALLOWED_CONFIGS[config],
                   "--variant", variant, "--seed", str(int(seed))]
            f = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
            job = {"id": jid, "config": config, "variant": variant, "seed": seed, "cmd": " ".join(cmd[1:]),
                   "log": str(log_path), "started": time.time(), "proc": proc, "_file": f}
            self.jobs[jid] = job
            return self.describe(job)

    def describe(self, job: dict) -> dict:
        code = job["proc"].poll()
        log_tail = ""
        try:
            log_tail = Path(job["log"]).read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            pass
        from sgerup.config import make_config
        from sgerup.engine import run_dir
        cfg = make_config(self.ALLOWED_CONFIGS[job["config"]], job["variant"], {"train": {"seed": job["seed"]}})
        st_path = run_dir(cfg) / "status.json"
        status = json.loads(st_path.read_text(encoding="utf-8")) if st_path.exists() else None
        return {k: v for k, v in job.items() if not k.startswith("_") and k != "proc"} | {
            "state": "running" if code is None else ("succeeded" if code == 0 else "failed"),
            "returncode": code, "log_tail": log_tail, "run_status": status}


model_state = ModelState()
results = ResultStore(settings.result_cache)
jobs = JobRunner()
_dataset_cache: dict[tuple, list] = {}
_dataset_lock = threading.Lock()


def dataset_samples(name: str, split: str | None):
    from sgerup.data.datasets import load_samples
    key = (name, split)
    with _dataset_lock:
        if key not in _dataset_cache:
            _dataset_cache[key] = load_samples(name, split)
        return _dataset_cache[key]


def invalidate_dataset(name: str) -> None:
    with _dataset_lock:
        for k in [k for k in _dataset_cache if k[0] == name]:
            _dataset_cache.pop(k)
