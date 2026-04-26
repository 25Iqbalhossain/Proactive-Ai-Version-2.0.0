from __future__ import annotations

import unittest

from benchmark.benchmark_engine import BenchmarkEngine


class BenchmarkModeFilteringTests(unittest.TestCase):
    def test_explicit_mode_rejects_both_feedback_algorithms(self):
        reason = BenchmarkEngine._check_compat(
            meta={"feedback": "both"},
            mode="explicit",
            n_users=100,
            n_items=100,
        )
        self.assertIn("incompatible with mode=explicit", reason)

    def test_implicit_mode_rejects_both_feedback_algorithms(self):
        reason = BenchmarkEngine._check_compat(
            meta={"feedback": "both"},
            mode="implicit",
            n_users=100,
            n_items=100,
        )
        self.assertIn("incompatible with mode=implicit", reason)

    def test_hybrid_mode_accepts_both_feedback_algorithms(self):
        reason = BenchmarkEngine._check_compat(
            meta={"feedback": "both"},
            mode="hybrid",
            n_users=100,
            n_items=100,
        )
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
