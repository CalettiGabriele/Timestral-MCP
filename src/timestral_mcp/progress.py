"""Avanzamento e tempi massimi dei tool lunghi, validi sia come Task MCP sia in esecuzione sincrona."""

from __future__ import annotations

import anyio
from fastmcp import Context
from fastmcp.dependencies import Progress

SYNC_SECONDS = 90  # client che non supportano i Task (es. Claude Desktop): restare entro i timeout dei client
TASK_SECONDS = 900  # come Task in background non c'è un client in attesa: si può lavorare più a lungo


def running_as_task() -> bool:
    """True se il tool sta girando come Task MCP in background (worker), False se è sincrono."""
    try:
        from docket.dependencies import current_execution

        current_execution.get()
        return True
    except (ImportError, LookupError):
        return False


def effective_seconds(requested: int | None) -> float:
    """Tempo massimo effettivo: quello richiesto, altrimenti 90 s sincrono o 900 s come Task."""
    return float(requested) if requested else float(TASK_SECONDS if running_as_task() else SYNC_SECONDS)


class Reporter:
    """Riporta l'avanzamento sia al Task (Progress) sia al client in esecuzione sincrona (notifiche MCP)."""

    def __init__(self, progress: Progress | None, ctx: Context | None) -> None:
        self._progress = progress
        self._ctx = ctx
        self._total = 0
        self._done = 0

    async def update(self, done: int, total: int, message: str) -> None:
        """Imposta lo stato (``done`` passi su ``total``) con un messaggio leggibile."""
        total = max(int(total), 1)
        try:
            if self._progress is not None:
                if total != self._total:
                    await self._progress.set_total(total)
                await self._progress.set_message(message)
                if done > self._done:
                    await self._progress.increment(done - self._done)
        except Exception:  # noqa: BLE001 - l'avanzamento non deve mai far fallire il calcolo
            pass
        self._total, self._done = total, max(done, self._done)
        if self._ctx is not None:
            try:
                await self._ctx.report_progress(done, total, message)
            except Exception:  # noqa: BLE001
                pass

    def from_thread(self):
        """Callback ``(done, total, message)`` richiamabile dal thread di calcolo."""

        def report(done: int, total: int, message: str) -> None:
            try:
                anyio.from_thread.run(self.update, done, total, message)
            except RuntimeError:  # fuori da un thread anyio (chiamata diretta)
                pass

        return report
