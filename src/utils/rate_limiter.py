import time
import threading
from collections import defaultdict, deque
from typing import Tuple


class RateLimiter:
    """Sliding-window limiter, in memory (resets on restart)."""

    def __init__(self, limit: int, window_s: int = 3600):
        self.limit = limit
        self.window_s = window_s
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False, int(self.window_s - (now - q[0])) + 1
            q.append(now)
            return True, 0