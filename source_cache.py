"""Small, thread-safe caches for data derived from chemistry source files.

The GUI's workers commonly ask several independent helpers for information
from the same source file.  A cache key includes the resolved path, nanosecond
mtime, and file size, so replacing or editing a file invalidates every derived
value without requiring callers to coordinate cache invalidation.
"""

from __future__ import annotations

import os
from threading import RLock
from typing import Callable, Hashable, TypeVar


T = TypeVar("T")


def file_cache_key(path: os.PathLike | str) -> tuple[str, int, int]:
    """Return a stable, automatically invalidating identity for *path*."""
    resolved = os.path.realpath(os.path.abspath(os.fspath(path)))
    stat = os.stat(resolved)
    return resolved, stat.st_mtime_ns, stat.st_size


class ComputationCache:
    """Memoize file-derived computations safely across GUI worker threads."""

    def __init__(self) -> None:
        self._values: dict[Hashable, object] = {}
        self._lock = RLock()

    def get(self, key: Hashable, factory: Callable[[], T]) -> T:
        # RLock is intentional: one cached factory may request another cached
        # value from the same source module (for example occupations -> S).
        with self._lock:
            if key in self._values:
                return self._values[key]  # type: ignore[return-value]
            value = factory()
            self._values[key] = value
            return value

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

