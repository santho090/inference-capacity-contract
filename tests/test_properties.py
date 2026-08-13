from __future__ import annotations

import random
import unittest
from dataclasses import replace
from decimal import Decimal

from inference_capacity_contract import (
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    ModelSpec,
    RuntimeKVCapacityPoint,
    RuntimeVariant,
    WorkloadProfile,
    capacity_for,
    recommend_scale,
)


def _model(**overrides: object) -> ModelSpec:
    values: dict[str, object] = {
        "model_id": "test/model",
        "revision": "sha256:test",
        "parameter_count": 1,
        "num_layers": 1,
        "num_kv_heads": 1,
        "head_dim": 1,
        "explicit_weight_bytes": 100,
        "max_model_len": None,
    }
    values.update(overrides)
    return ModelSpec(**values)  # type: ignore[arg-type]


def _runtime(**overrides: object) -> RuntimeVariant:
    values: dict[str, object] = {
        "engine": "vllm",
        "version": "test",
        "block_size_tokens": 4,
        "runtime_overhead_bytes_per_device": 0,
        "activation_reserve_bytes_per_device": 0,
        "max_num_seqs": None,
    }
    values.update(overrides)
    return RuntimeVariant(**values)  # type: ignore[arg-type]


class CapacityPropertyTests(unittest.TestCase):
    def test_int4_weight_storage_rounds_up_to_a_complete_byte(self) -> None:
        model = _model(
            parameter_count=3,
            weight_dtype="int4",
            explicit_weight_bytes=None,
        )
        self.assertEqual(model.weight_bytes, 2)

    def test_tensor_parallel_weight_division_uses_integer_ceiling(self) -> None:
        total_weight_bytes = 2**60 + 1
        contract = capacity_for(
            _model(explicit_weight_bytes=total_weight_bytes),
            HardwareSpec("large-x3", "nvidia", 3, 2**61, memory_utilization_limit=1.0),
            _runtime(tensor_parallel_size=3, kv_heads_per_device=1),
        )
        self.assertEqual(contract.memory_bytes_per_device["weights"], (total_weight_bytes + 2) // 3)
        self.assertIn("assume even sharding", " ".join(contract.warnings))

    def test_memory_utilization_uses_decimal_floor_without_binary_float_rounding(self) -> None:
        memory_bytes = 2**60 + 1
        contract = capacity_for(
            _model(),
            HardwareSpec("large", "nvidia", 1, memory_bytes, memory_utilization_limit=0.9),
            _runtime(),
        )
        expected = int(Decimal(memory_bytes) * Decimal("0.9"))
        self.assertEqual(contract.memory_bytes_per_device["usable_budget"], expected)

    def test_explicit_per_device_weight_bytes_replace_even_sharding_assumption(self) -> None:
        contract = capacity_for(
            _model(explicit_weight_bytes=1_000),
            HardwareSpec("tp2", "nvidia", 2, 10_000, memory_utilization_limit=1.0),
            _runtime(
                tensor_parallel_size=2,
                kv_heads_per_device=1,
                weight_bytes_per_device_override=700,
            ),
        )
        self.assertEqual(contract.memory_bytes_per_device["weights"], 700)
        self.assertNotIn("assume even sharding", " ".join(contract.warnings))

    def test_custom_and_sub_byte_layouts_require_exact_runtime_envelopes(self) -> None:
        exact = capacity_for(
            _model(attention_type="custom"),
            HardwareSpec("custom", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
            _runtime(
                kv_capacity_hardware_id="custom",
                kv_capacity_memory_bytes_per_device=9900,
                kv_capacity_envelope_override=(RuntimeKVCapacityPoint(100, 10),),
            ),
        )
        self.assertIsNone(exact.kv_bytes_per_token_per_device)
        self.assertEqual(exact.max_sequences_at(100), 10)

        with self.assertRaisesRegex(ContractError, "runtime KV capacity envelope"):
            capacity_for(
                _model(attention_type="custom", kv_bytes_per_token_per_device_override=123),
                HardwareSpec("custom", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
                _runtime(),
            )

        with self.assertRaisesRegex(ContractError, "sub-byte"):
            capacity_for(
                _model(kv_bytes_per_token_per_device_override=321),
                HardwareSpec("int4", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
                _runtime(kv_cache_dtype="int4"),
            )

    def test_kv_heads_per_device_cannot_exceed_model_layout(self) -> None:
        with self.assertRaisesRegex(ContractError, "cannot exceed"):
            capacity_for(
                _model(num_kv_heads=2),
                HardwareSpec("tp2", "nvidia", 2, 10_000, memory_utilization_limit=1.0),
                _runtime(tensor_parallel_size=2, kv_heads_per_device=3),
            )

    def test_kv_heads_per_device_must_cover_the_replica_layout(self) -> None:
        with self.assertRaisesRegex(ContractError, "under-represents"):
            capacity_for(
                _model(num_kv_heads=8),
                HardwareSpec("tp2", "nvidia", 2, 10_000, memory_utilization_limit=1.0),
                _runtime(tensor_parallel_size=2, kv_heads_per_device=1),
            )

    def test_exactly_one_kv_block_fits_and_one_byte_less_does_not(self) -> None:
        # KV bytes/token = 2 (K and V) * 1 layer * 1 head * 1 dim * 2 BF16 bytes = 4.
        # Four tokens per block therefore require exactly 16 bytes.
        exact = capacity_for(
            _model(),
            HardwareSpec("exact", "nvidia", 1, 116, memory_utilization_limit=1.0),
            _runtime(),
        )
        short = capacity_for(
            _model(),
            HardwareSpec("short", "nvidia", 1, 115, memory_utilization_limit=1.0),
            _runtime(),
        )
        self.assertTrue(exact.fits)
        self.assertEqual(exact.kv_capacity_blocks_per_device, 1)
        self.assertEqual(exact.max_context_tokens, 4)
        self.assertEqual(exact.max_sequences_at(1), 1)
        self.assertFalse(short.fits)
        self.assertEqual(short.kv_capacity_blocks_per_device, 0)

    def test_randomized_results_match_an_independent_integer_oracle(self) -> None:
        rng = random.Random(7)
        dtype_bytes = {"bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}
        for case in range(250):
            layers = rng.randint(1, 80)
            heads = rng.randint(1, 32)
            head_dim = rng.choice((32, 64, 128, 256))
            dtype = rng.choice(tuple(dtype_bytes))
            block_size = rng.choice((1, 4, 8, 16, 32))
            weights = rng.randint(1, 50_000_000)
            runtime_reserve = rng.randint(0, 5_000_000)
            activation_reserve = rng.randint(0, 5_000_000)
            memory = rng.randint(1, 100_000_000)
            model_limit = rng.choice((None, rng.randint(1, 100_000)))

            contract = capacity_for(
                _model(
                    num_layers=layers,
                    num_kv_heads=heads,
                    head_dim=head_dim,
                    explicit_weight_bytes=weights,
                    max_model_len=model_limit,
                ),
                HardwareSpec(f"random-{case}", "nvidia", 1, memory, memory_utilization_limit=1.0),
                _runtime(
                    kv_cache_dtype=dtype,
                    block_size_tokens=block_size,
                    runtime_overhead_bytes_per_device=runtime_reserve,
                    activation_reserve_bytes_per_device=activation_reserve,
                ),
            )

            kv_bytes_per_token = 2 * layers * heads * head_dim * dtype_bytes[dtype]
            kv_block_bytes = kv_bytes_per_token * block_size
            kv_available = max(0, memory - weights - runtime_reserve - activation_reserve)
            expected_blocks = kv_available // kv_block_bytes
            expected_tokens = expected_blocks * block_size
            expected_fit = memory - weights - runtime_reserve - activation_reserve >= kv_block_bytes
            expected_context = expected_tokens if model_limit is None else min(expected_tokens, model_limit)
            if not expected_fit:
                expected_context = 0

            self.assertEqual(contract.kv_bytes_per_token_per_device, kv_bytes_per_token)
            self.assertEqual(contract.kv_capacity_blocks_per_device, expected_blocks)
            self.assertEqual(contract.kv_capacity_tokens_per_device, expected_tokens)
            self.assertEqual(contract.fits, expected_fit)
            self.assertEqual(contract.max_context_tokens, expected_context)
            if expected_fit:
                context = rng.randint(1, expected_context)
                blocks_per_sequence = (context + block_size - 1) // block_size
                self.assertEqual(contract.max_sequences_at(context), expected_blocks // blocks_per_sequence)

    def test_capacity_is_monotone_in_memory_and_reserves(self) -> None:
        model = _model(max_model_len=100_000)
        runtime = _runtime(runtime_overhead_bytes_per_device=100, activation_reserve_bytes_per_device=100)
        smaller = capacity_for(
            model,
            HardwareSpec("small", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
            runtime,
        )
        larger = capacity_for(
            model,
            HardwareSpec("large", "nvidia", 1, 20_000, memory_utilization_limit=1.0),
            runtime,
        )
        larger_reserve = capacity_for(
            model,
            HardwareSpec("large", "nvidia", 1, 20_000, memory_utilization_limit=1.0),
            replace(runtime, activation_reserve_bytes_per_device=1_000),
        )
        smaller_blocks = smaller.kv_capacity_blocks_per_device
        larger_blocks = larger.kv_capacity_blocks_per_device
        reserve_blocks = larger_reserve.kv_capacity_blocks_per_device
        assert smaller_blocks is not None and larger_blocks is not None and reserve_blocks is not None
        self.assertGreaterEqual(larger_blocks, smaller_blocks)
        self.assertGreaterEqual(larger.max_context_tokens, smaller.max_context_tokens)
        self.assertLessEqual(reserve_blocks, larger_blocks)

    def test_sequence_capacity_is_monotone_in_context_and_honors_runtime_cap(self) -> None:
        contract = capacity_for(
            _model(max_model_len=128),
            HardwareSpec("bounded", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
            _runtime(max_num_seqs=7),
        )
        capacities = [contract.max_sequences_at(context) for context in range(1, 129)]
        self.assertTrue(all(left >= right for left, right in zip(capacities, capacities[1:], strict=False)))
        self.assertTrue(all(value <= 7 for value in capacities))
        self.assertEqual(contract.max_sequences_at(129), 0)

    def test_custom_context_points_are_sorted_deduplicated_and_bounded(self) -> None:
        args = (
            _model(max_model_len=64),
            HardwareSpec("custom", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
            _runtime(),
        )
        contract = capacity_for(*args, context_points=(32, 1, 32, 17))
        self.assertEqual([point.context_tokens for point in contract.concurrency_envelope], [1, 17, 32])
        with self.assertRaisesRegex(ContractError, "positive"):
            capacity_for(*args, context_points=(0,))
        with self.assertRaisesRegex(ContractError, "max_context_tokens"):
            capacity_for(*args, context_points=(65,))
        with self.assertRaisesRegex(ContractError, "cannot be empty"):
            capacity_for(*args, context_points=())

    def test_scaling_rejects_concurrency_context_outside_contract(self) -> None:
        contract = capacity_for(
            _model(max_model_len=64),
            HardwareSpec("bounded", "nvidia", 1, 10_000, memory_utilization_limit=1.0),
            _runtime(),
        )
        profile = WorkloadProfile(
            profile_id="outside-context",
            model_id="test/model",
            model_revision="sha256:test",
            hardware_id="bounded",
            runtime_engine="vllm",
            runtime_version="test",
            request_rate_per_second=0,
            sustainable_concurrent_sequences_per_replica=1,
            peak_concurrent_sequences=1,
            concurrency_context_tokens=65,
            evidence=(
                EvidenceRecord(
                    evidence_id="measured",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://measured",
                    scope="exact variant",
                ),
            ),
        )
        with self.assertRaisesRegex(ContractError, "supported context"):
            recommend_scale(contract, profile)


if __name__ == "__main__":
    unittest.main()
