"""Configurazione da variabili d'ambiente (prefisso ``TIMESTRAL_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_SIZES = ("5M", "28M", "123M", "209M")
MODEL_REPO = "mistralai/Timestral-{size}"
MODEL_WEIGHTS_MB = {"5M": 21, "28M": 111, "123M": 490, "209M": 837}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "si", "on"}


@dataclass(frozen=True)
class Settings:
    default_model: str = "209M"
    device: str = "cpu"
    preload: bool = True
    export_dir: Path = Path.home() / "TimestralMCP" / "exports"
    skills_dir: Path | None = None
    max_rows: int = 2_000_000
    max_context: int = 32_768
    model_horizon_block: int = 1_024
    max_datasets: int = 20
    max_forecasts: int = 100

    @classmethod
    def from_env(cls) -> Settings:
        model = os.environ.get("TIMESTRAL_DEFAULT_MODEL", cls.default_model).upper()
        if model not in MODEL_SIZES:
            raise ValueError(
                f"TIMESTRAL_DEFAULT_MODEL={model!r} non valido: scegli tra {', '.join(MODEL_SIZES)}"
            )
        export_dir = Path(os.environ.get("TIMESTRAL_EXPORT_DIR", cls.export_dir)).expanduser()
        skills = os.environ.get("TIMESTRAL_SKILLS_DIR")
        return cls(
            default_model=model,
            device=os.environ.get("TIMESTRAL_DEVICE", cls.device),
            preload=_env_bool("TIMESTRAL_PRELOAD", cls.preload),
            export_dir=export_dir,
            skills_dir=Path(skills).expanduser() if skills else None,
            max_rows=int(os.environ.get("TIMESTRAL_MAX_ROWS", cls.max_rows)),
        )


def get_settings() -> Settings:
    return Settings.from_env()
