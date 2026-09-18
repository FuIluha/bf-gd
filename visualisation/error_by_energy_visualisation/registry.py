"""Small in-process registry keeping large NumPy sessions out of the browser."""

from dataclasses import dataclass
from threading import RLock
import time

from .session import ExplorerSession


@dataclass
class _Entry:
    session: ExplorerSession
    last_seen: float


class SessionRegistry:
    def __init__(self, dataset, seed, workers, max_sessions=16):
        self.dataset = dataset
        self.seed = int(seed)
        self.workers = max(1, int(workers))
        self.max_sessions = max(1, int(max_sessions))
        self._entries = {}
        self._lock = RLock()

    def get(self, session_id):
        if not session_id:
            raise ValueError("browser session id is missing")
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                self._evict_if_needed()
                entry = _Entry(
                    session=ExplorerSession(
                        graph=self.dataset.graph,
                        batch=self.dataset.batch,
                        seed=self.seed,
                        workers=self.workers,
                        metadata=self.dataset.metadata,
                    ),
                    last_seen=time.monotonic(),
                )
                self._entries[session_id] = entry
            entry.last_seen = time.monotonic()
            return entry.session

    def replace(self, session_id, session):
        with self._lock:
            self._entries[session_id] = _Entry(
                session=session,
                last_seen=time.monotonic(),
            )

    def _evict_if_needed(self):
        if len(self._entries) < self.max_sessions:
            return
        oldest = min(
            self._entries,
            key=lambda key: self._entries[key].last_seen,
        )
        del self._entries[oldest]
