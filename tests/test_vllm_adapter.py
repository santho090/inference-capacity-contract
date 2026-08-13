from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from inference_capacity_contract import ContractError, import_vllm_runtime_snapshot

ROOT = Path(__file__).parents[1]


def _snapshot() -> dict[str, object]:
    value = json.loads((ROOT / "docs" / "fixtures" / "vllm-runtime-snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("snapshot fixture must contain an object")
    return value


class VLLMAdapterTests(unittest.TestCase):
    def test_imports_exact_worker_budget_and_group_capacity(self) -> None:
        profile = import_vllm_runtime_snapshot(_snapshot(), source="benchmark://runtime-snapshot")

        self.assertEqual(profile.memory_budget_bytes_per_device, 296868139500)
        self.assertEqual(profile.runtime_overhead_bytes_per_device, 2 * 1024**3)
        self.assertEqual(profile.kv_capacity_envelope[0].context_tokens, 1_048_576)
        self.assertEqual(profile.kv_capacity_envelope[0].max_sequences, 24)
        self.assertEqual(profile.evidence.metrics["limiting_worker_rank"], 0)
        self.assertEqual(profile.evidence.metrics["worker_count"], 8)
        self.assertTrue(str(profile.evidence.metrics["snapshot_digest"]).startswith("sha256:"))

    def test_uses_the_limiting_worker_without_mixing_rank_ledgers(self) -> None:
        snapshot = _snapshot()
        workers = snapshot["workers"]
        assert isinstance(workers, list)
        limiting = workers[-1]
        assert isinstance(limiting, dict)
        limiting["available_kv_cache_memory_bytes"] = int(limiting["available_kv_cache_memory_bytes"]) - 1024

        profile = import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        self.assertEqual(profile.evidence.metrics["limiting_worker_rank"], 7)
        self.assertEqual(profile.kv_capacity_memory_bytes_per_device, 226001178092)
        self.assertEqual(profile.runtime_overhead_bytes_per_device, 2 * 1024**3 + 1024)
        self.assertEqual(profile.memory_budget_bytes_per_device, 296868139500)

    def test_caps_memory_concurrency_at_the_scheduler_limit(self) -> None:
        snapshot = _snapshot()
        context = snapshot["context_tokens"]
        assert isinstance(context, int)
        snapshot["kv_cache_max_concurrency"] = 30.75
        snapshot["kv_cache_size_tokens"] = int(30.75 * context)

        profile = import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        self.assertEqual(profile.kv_capacity_envelope[0].max_sequences, 24)

    def test_rejects_incomplete_or_inconsistent_snapshots(self) -> None:
        cases: tuple[tuple[str, object, str], ...] = (
            ("workers", [], "non-empty array"),
            ("kv_cache_size_tokens", 1, "does not match"),
            ("kv_cache_max_concurrency", math.inf, "finite positive number"),
        )
        for field, value, error in cases:
            with self.subTest(field=field):
                snapshot = _snapshot()
                snapshot[field] = value
                with self.assertRaisesRegex(ContractError, error):
                    import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        snapshot = _snapshot()
        workers = snapshot["workers"]
        assert isinstance(workers, list) and isinstance(workers[-1], dict)
        workers[-1]["device_memory_bytes"] = int(workers[-1]["device_memory_bytes"]) - 1
        with self.assertRaisesRegex(ContractError, "different device memory sizes"):
            import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        snapshot = _snapshot()
        workers = snapshot["workers"]
        assert isinstance(workers, list) and isinstance(workers[0], dict)
        workers[0]["requested_memory_bytes"] = int(workers[0]["requested_memory_bytes"]) - 1
        with self.assertRaisesRegex(ContractError, "does not match device memory"):
            import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        snapshot = _snapshot()
        workers = snapshot["workers"]
        assert isinstance(workers, list) and isinstance(workers[0], dict)
        workers[0]["model_memory_usage_bytes"] = int(workers[0]["requested_memory_bytes"])
        with self.assertRaisesRegex(ContractError, "components exceed"):
            import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")

        snapshot = _snapshot()
        snapshot["surprise"] = True
        with self.assertRaisesRegex(ContractError, "unknown field"):
            import_vllm_runtime_snapshot(snapshot, source="benchmark://runtime-snapshot")


if __name__ == "__main__":
    unittest.main()
