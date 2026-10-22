"""Server settings from environment variables (see .env.example). Secrets never leave the server."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # optional
    pass


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    device: str = os.getenv("SGERUP_DEVICE", "auto")
    detector_weights: str = os.getenv("SGERUP_DETECTOR_WEIGHTS", "weights/yolov3u.pt")
    imgsz: int = int(os.getenv("SGERUP_IMGSZ", "448"))
    max_upload_mb: float = float(os.getenv("SGERUP_MAX_UPLOAD_MB", "15"))
    result_cache: int = int(os.getenv("SGERUP_RESULT_CACHE", "40"))
    cors_origins: list[str] = field(default_factory=lambda: [o.strip() for o in os.getenv(
        "SGERUP_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()])
    allow_training_jobs: bool = _bool("SGERUP_ALLOW_TRAINING_JOBS", True)
    preload_detector: bool = _bool("SGERUP_PRELOAD_DETECTOR", True)
    # Ask AI
    llm_model: str = os.getenv("SGERUP_LLM_MODEL", "claude-opus-5-5")
    llm_enabled: str = os.getenv("SGERUP_LLM", "auto")  # auto | on | off
    llm_effort: str = os.getenv("SGERUP_LLM_EFFORT", "low")
    llm_max_tokens: int = int(os.getenv("SGERUP_LLM_MAX_TOKENS", "16000"))

    @property
    def torch_device(self) -> str | None:
        return None if self.device == "auto" else self.device

    def llm_configured(self) -> tuple[bool, str]:
        """LLM mode requires explicit server-side credentials; otherwise local fallback mode."""
        if self.llm_enabled == "off":
            return False, "disabled by SGERUP_LLM=off"
        has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        if self.llm_enabled == "on" or has_key:
            try:
                import anthropic  # noqa: F401
            except ImportError:
                return False, "the 'anthropic' package is not installed"
            return True, "credentials configured on the server" if has_key else "SGERUP_LLM=on (SDK credential chain)"
        return False, "no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN set on the server"


settings = Settings()
