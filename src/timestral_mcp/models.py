"""Caricamento lazy e cache dei modelli Timestral."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from timestral_mcp.config import MODEL_REPO, MODEL_SIZES, MODEL_WEIGHTS_MB, get_settings

ForecasterFactory = Callable[[str], Any]

_lock = threading.Lock()
_cache: dict[str, Any] = {}
_factory: ForecasterFactory | None = None


def _default_factory(size: str) -> Any:
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(MODEL_REPO.format(size=size), device=get_settings().device)


def set_factory(factory: ForecasterFactory | None) -> None:
    """Sostituisce il modo di costruire i forecaster (usato dai test) e svuota la cache."""
    global _factory
    with _lock:
        _factory = factory
        _cache.clear()


def normalize_size(size: str | None) -> str:
    size = (size or get_settings().default_model).upper().replace("TIMESTRAL-", "")
    if size not in MODEL_SIZES:
        raise ValueError(f"Modello {size!r} non disponibile: scegli tra {', '.join(MODEL_SIZES)}")
    return size


def get_forecaster(size: str | None = None) -> Any:
    size = normalize_size(size)
    with _lock:
        if size not in _cache:
            _cache[size] = (_factory or _default_factory)(size)
        return _cache[size]


def is_loaded(size: str) -> bool:
    return size in _cache


class ModelNotDownloadedError(ValueError):
    """Il modello richiesto non è sul disco e il download non è stato autorizzato."""


def is_cached(size: str) -> bool:
    """True se i pesi sono già nella cache di Hugging Face (nessun download necessario)."""
    if _factory is not None:  # factory personalizzata (test): non scarica nulla
        return True
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(MODEL_REPO.format(size=size), "model.safetensors")
    return isinstance(found, str)


def cached_sizes() -> list[str]:
    return [s for s in MODEL_SIZES if is_cached(s)]


def ensure_available(size: str, allow_download: bool = False) -> None:
    """Impedisce download silenziosi: senza consenso esplicito solleva un errore con le alternative."""
    if is_loaded(size) or allow_download or is_cached(size):
        return
    available = cached_sizes()
    alt = (
        f" Modelli già disponibili: {', '.join(available)}."
        if available
        else " Nessun modello è ancora scaricato."
    )
    raise ModelNotDownloadedError(
        f"Il modello Timestral-{size} non è ancora scaricato (circa {MODEL_WEIGHTS_MB[size]} MB: "
        f"il download può richiedere diversi minuti).{alt} Chiedi all'utente se vuole scaricarlo e, se conferma, "
        f"ripeti la chiamata con allow_download=true; oppure usa un modello già disponibile. "
        f"Da terminale: uv run timestral-mcp download {size}"
    )


def download(size: str) -> str:
    """Scarica i pesi nella cache di Hugging Face (usato dal comando `timestral-mcp download`)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_REPO.format(size=normalize_size(size)))
