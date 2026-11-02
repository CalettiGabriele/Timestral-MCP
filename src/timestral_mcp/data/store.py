"""Archivio in memoria di dataset e previsioni (con limite di capienza)."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict


class Store[T]:
    def __init__(self, prefix: str, capacity: int) -> None:
        self._prefix = prefix
        self._capacity = capacity
        self._items: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return f"{self._prefix}_{uuid.uuid4().hex[:8]}"

    def put(self, key: str, item: T) -> None:
        with self._lock:
            self._items[key] = item
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def get(self, key: str) -> T:
        with self._lock:
            if key not in self._items:
                known = ", ".join(self._items) or "nessuno"
                raise KeyError(
                    f"{key!r} non trovato (disponibili: {known}). Potrebbe essere stato eliminato per fare spazio."
                )
            self._items.move_to_end(key)
            return self._items[key]

    def items(self) -> list[tuple[str, T]]:
        with self._lock:
            return list(self._items.items())

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
