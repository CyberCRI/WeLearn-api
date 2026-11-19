import threading
import unittest
from types import SimpleNamespace

from src.app.shared.infra import rate_limit
from src.app.shared.infra.rate_limit import (
    SlidingWindowRateLimiter,
    client_ip,
    reset_rate_limiters,
)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SlidingWindowRateLimiterTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.limiter = SlidingWindowRateLimiter(3, 60, clock=self.clock)

    def test_allows_up_to_the_limit_then_blocks(self):
        self.assertEqual([self.limiter.hit("a") for _ in range(3)], [0, 0, 0])

        retry_after = self.limiter.hit("a")

        self.assertGreater(retry_after, 0)
        self.assertLessEqual(retry_after, 60)

    def test_keys_are_independent(self):
        for _ in range(3):
            self.limiter.hit("a")

        self.assertEqual(self.limiter.hit("b"), 0)

    def test_window_slides(self):
        self.limiter.hit("a")
        self.clock.now += 30
        self.limiter.hit("a")
        self.limiter.hit("a")
        self.assertGreater(self.limiter.hit("a"), 0)

        # the first hit leaves the window, one more is allowed
        self.clock.now += 30.5
        self.assertEqual(self.limiter.hit("a"), 0)
        self.assertGreater(self.limiter.hit("a"), 0)

    def test_retry_after_is_when_the_oldest_hit_expires(self):
        for _ in range(3):
            self.limiter.hit("a")
        self.clock.now += 45

        self.assertAlmostEqual(self.limiter.hit("a"), 15)

    def test_blocked_hits_are_not_counted(self):
        for _ in range(3):
            self.limiter.hit("a")
        for _ in range(10):
            self.limiter.hit("a")

        self.clock.now += 60
        self.assertEqual(self.limiter.hit("a"), 0)

    def test_memory_is_bounded(self):
        limiter = SlidingWindowRateLimiter(1, 60, max_keys=5, clock=self.clock)
        for i in range(20):
            limiter.hit(f"k{i}")

        self.assertEqual(len(limiter), 5)
        # the most recent keys are kept
        self.assertGreater(limiter.hit("k19"), 0)

    def test_stale_keys_are_swept(self):
        for i in range(10):
            self.limiter.hit(f"k{i}")
        self.clock.now += 120

        for _ in range(rate_limit._SWEEP_EVERY):
            self.limiter.hit("fresh")

        self.assertEqual(len(self.limiter), 1)

    def test_reset_rate_limiters(self):
        for _ in range(3):
            self.limiter.hit("a")

        reset_rate_limiters()

        self.assertEqual(len(self.limiter), 0)
        self.assertEqual(self.limiter.hit("a"), 0)

    def test_thread_safe(self):
        limiter = SlidingWindowRateLimiter(50, 60)
        results = []

        def worker():
            for _ in range(20):
                results.append(limiter.hit("shared"))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(1 for r in results if r == 0), 50)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(0, 60)
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(1, 0)


class ClientIpTests(unittest.TestCase):
    def test_peer_address(self):
        request = SimpleNamespace(client=SimpleNamespace(host="10.0.0.7"))
        self.assertEqual(client_ip(request), "10.0.0.7")

    def test_no_peer(self):
        self.assertEqual(client_ip(SimpleNamespace(client=None)), "unknown")


class InFlightLimiterTests(unittest.TestCase):
    def test_up_to_the_limit_per_key_until_released(self):
        limiter = rate_limit.InFlightLimiter(2)

        self.assertTrue(limiter.acquire("a"))
        self.assertTrue(limiter.acquire("a"))
        self.assertFalse(limiter.acquire("a"))
        # another key has its own places
        self.assertTrue(limiter.acquire("b"))

        limiter.release("a")
        self.assertEqual(limiter.running("a"), 1)
        self.assertTrue(limiter.acquire("a"))

    def test_a_key_with_nothing_running_is_forgotten(self):
        limiter = rate_limit.InFlightLimiter(1)
        limiter.acquire("a")
        limiter.release("a")
        # an extra release does not go below nothing
        limiter.release("a")

        self.assertEqual(len(limiter), 0)
        self.assertEqual(limiter.running("a"), 0)

    def test_reset_with_the_other_limiters(self):
        limiter = rate_limit.InFlightLimiter(1)
        limiter.acquire("a")

        reset_rate_limiters()

        self.assertTrue(limiter.acquire("a"))

    def test_needs_a_positive_limit(self):
        with self.assertRaises(ValueError):
            rate_limit.InFlightLimiter(0)
