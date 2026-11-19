# src/app/shared/infra/rate_limit.py
"""
Small in-memory sliding-window rate limiter.

Limits are kept in the memory of the process: they apply **per pod/process**,
are lost on restart, and multiply by the number of replicas (or uvicorn
workers). They are a brake against floods from one client, not a global quota.

Client IP: `client_ip()` uses `request.client.host`. Uvicorn sets it from
`X-Forwarded-For` only when the peer is listed in the FORWARDED_ALLOW_IPS env
var (default 127.0.0.1); it then keeps the rightmost address that is not
trusted, i.e. the one the proxy added, so a client cannot forge it. Behind the
k8s (Traefik) ingress, set FORWARDED_ALLOW_IPS to the CIDR of the Traefik pods:
without it, `client_ip()` is the ingress' address, every user shares the same
"IP" bucket and a per-IP limit is only a coarse ceiling.
"""

import threading
import time
from collections import deque
from collections.abc import Callable
from weakref import WeakSet

from fastapi import Request

# drop keys that have no hit left in their window every this many hits
_SWEEP_EVERY = 256

_limiters: "WeakSet[SlidingWindowRateLimiter | InFlightLimiter]" = WeakSet()


class SlidingWindowRateLimiter:
    """
    Allows `limit` hits per key in any `window_seconds` window. Thread-safe.

    Memory is bounded: stale keys are swept regularly, and past `max_keys` the
    least recently hit keys are dropped (which only makes the limiter more
    lenient for them).
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ):
        if limit < 1 or window_seconds <= 0 or max_keys < 1:
            raise ValueError("limit, window_seconds and max_keys must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._calls = 0
        _limiters.add(self)

    def hit(self, key: str) -> float:
        """
        Record a hit for `key` if it is under the limit and return 0. Otherwise
        record nothing and return the seconds until the next hit is allowed.
        """
        with self._lock:
            now = self._clock()
            self._calls += 1
            if self._calls % _SWEEP_EVERY == 0:
                self._sweep(now)

            hits = self._hits.pop(key, None) or deque()
            self._expire(hits, now)
            # re-inserted last: dict order is least recently hit first
            self._hits[key] = hits

            if len(hits) >= self.limit:
                return max(hits[0] + self.window_seconds - now, 0.001)

            hits.append(now)
            while len(self._hits) > self.max_keys:
                del self._hits[next(iter(self._hits))]
            return 0.0

    def peek(self, key: str) -> tuple[int, float]:
        """
        Without recording a hit: how many hits `key` has left, and the seconds
        until its oldest hit leaves the window (0 when it has none).
        """
        with self._lock:
            now = self._clock()
            hits = self._hits.get(key)
            if not hits:
                return self.limit, 0.0
            self._expire(hits, now)
            if not hits:
                return self.limit, 0.0
            return (
                max(self.limit - len(hits), 0),
                max(hits[0] + self.window_seconds - now, 0.0),
            )

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
            self._calls = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._hits)

    def _expire(self, hits: deque[float], now: float) -> None:
        while hits and hits[0] <= now - self.window_seconds:
            hits.popleft()

    def _sweep(self, now: float) -> None:
        for key in list(self._hits):
            hits = self._hits[key]
            self._expire(hits, now)
            if not hits:
                del self._hits[key]


class InFlightLimiter:
    """
    At most `limit` requests of one key running at once. Thread-safe; in the memory
    of the process, like the sliding window above.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._running: dict[str, int] = {}
        self._lock = threading.Lock()
        _limiters.add(self)

    def acquire(self, key: str) -> bool:
        """Count one more running request for `key`, or False when it has `limit`."""
        with self._lock:
            running = self._running.get(key, 0)
            if running >= self.limit:
                return False
            self._running[key] = running + 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            running = self._running.get(key, 0) - 1
            if running > 0:
                self._running[key] = running
            else:
                self._running.pop(key, None)

    def running(self, key: str) -> int:
        with self._lock:
            return self._running.get(key, 0)

    def reset(self) -> None:
        with self._lock:
            self._running.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._running)


def reset_rate_limiters() -> None:
    """Forget every hit of every limiter (tests)."""
    for limiter in list(_limiters):
        limiter.reset()


def client_ip(request: Request) -> str:
    """The peer address of the request; see the module docstring for its limits."""
    return request.client.host if request.client else "unknown"
