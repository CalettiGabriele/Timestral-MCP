"""Descrizione dei valori futuri delle covariate (es. promozioni pianificate)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_RANGES = 10


def fmt_date(ts: pd.Timestamp, freq: str) -> str:
    return f"{ts:%Y-%m-%d %H:%M}" if freq.lower().endswith(("h", "min")) else f"{ts:%Y-%m-%d}"


def _num(x: float) -> float:
    return round(float(x), 4)


def _ranges(index: pd.DatetimeIndex, active: np.ndarray, freq: str) -> list[str]:
    """Intervalli consecutivi di periodi attivi, come 'AAAA-MM-GG..AAAA-MM-GG'."""
    pos = np.flatnonzero(active)
    if pos.size == 0:
        return []
    groups = np.split(pos, np.flatnonzero(np.diff(pos) > 1) + 1)
    out = []
    for g in groups:
        a, b = fmt_date(index[g[0]], freq), fmt_date(index[g[-1]], freq)
        out.append(a if a == b else f"{a}..{b}")
    return out


def describe_future_values(frame: pd.DataFrame, freq: str) -> dict[str, dict[str, Any]]:
    """Per ogni colonna: se è un indicatore 0/1 gli intervalli attivi, altrimenti min/media/max."""
    out: dict[str, dict[str, Any]] = {}
    for col in frame.columns:
        s = frame[col]
        vals = s.dropna().to_numpy(dtype=float)
        if vals.size == 0:
            out[col] = {"type": "empty", "periods": int(len(s))}
        elif set(np.unique(vals)) <= {0.0, 1.0}:
            active = (s.fillna(0) == 1).to_numpy()
            ranges = _ranges(frame.index, active, freq)
            out[col] = {
                "type": "binary",
                "periods": int(len(s)),
                "active_periods": int(active.sum()),
                "active_ranges": ranges[:MAX_RANGES],
                "more_ranges": max(len(ranges) - MAX_RANGES, 0),
            }
        else:
            out[col] = {
                "type": "numeric",
                "periods": int(len(s)),
                "min": _num(vals.min()),
                "mean": _num(vals.mean()),
                "max": _num(vals.max()),
                "first": _num(vals[0]),
                "last": _num(vals[-1]),
            }
    return out


def format_future_values(desc: dict[str, dict[str, Any]]) -> list[str]:
    lines = []
    for col, d in desc.items():
        if d["type"] == "binary":
            if d["active_periods"] == 0:
                lines.append(f"{col}: sempre 0 (mai attiva) nei {d['periods']} periodi futuri.")
            else:
                extra = f" (+{d['more_ranges']} altri intervalli)" if d["more_ranges"] else ""
                lines.append(
                    f"{col}: attiva (=1) in {d['active_periods']} dei {d['periods']} periodi futuri: "
                    f"{', '.join(d['active_ranges'])}{extra}."
                )
        elif d["type"] == "numeric":
            lines.append(
                f"{col}: valori futuri tra {d['min']:g} e {d['max']:g} (media {d['mean']:g}; "
                f"primo {d['first']:g}, ultimo {d['last']:g})."
            )
        else:
            lines.append(f"{col}: nessun valore futuro.")
    return lines


def future_block(df: pd.DataFrame, freq: str) -> dict[str, Any] | None:
    """Righe dopo la fine di una colonna che ha valori più avanti (covariate note in anticipo)."""
    end = df.index.max()
    early = [i for c in df.columns if (i := df[c].last_valid_index()) is not None and i < end]
    if not early:
        return None
    block = df.iloc[df.index.get_loc(min(early)) + 1 :]
    cols = [c for c in df.columns if block[c].notna().any()]
    if block.empty or not cols:
        return None
    desc = describe_future_values(block[cols], freq)
    return {
        "from": fmt_date(block.index[0], freq),
        "to": fmt_date(block.index[-1], freq),
        "periods": int(len(block)),
        "columns": desc,
        "summary": format_future_values(desc),
    }
