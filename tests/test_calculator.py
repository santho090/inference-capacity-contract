from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any

from inference_capacity_contract import (
    CapacityContract,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareInventory,
    HardwareSpec,
    KVCapacityMode,
    ModelSpec,
    RuntimeKVCapacityPoint,
    RuntimeVariant,
    ValidationLevel,
    WorkloadProfile,
    capacity_for,
    recommend_scale,
    to_llmd_planner_payload,
    to_scaling_policy_input,
    what_fits,
)

GIB = 1024**3
BASE_MODEL = ModelSpec(
    model_id="example/7b",
    revision="sha256:example",
    parameter_count=7_000_000_000,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    weight_dtype="bf16",
    max_model_len=8192,
)
BASE_RUNTIME = RuntimeVariant(
    engine="vllm",
    version="0.8.5",
    runtime_overhead_bytes_per_device=1 * GIB,
    activation_reserve_bytes_per_device=2 * GIB,
    max_num_seqs=128,
)


def model_7b(**overrides: Any) -> ModelSpec:
    return replace(BASE_MODEL, **overrides)


def runtime(**overrides: Any) -> RuntimeVariant:
    return replace(BASE_RUNTIME, **overrides)


class CalculatorTests(unittest.TestCase):
    def test_analytical_fit_accounts_for_weights_reserves_and_kv(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        self.assertTrue(contract.fits)
        self.assertEqual(contract.schema_version, "capacity-contract-3.0")
        self.assertEqual(
            contract.validation_level,
            ValidationLevel.ANALYTICALLY_FEASIBLE,
        )
        self.assertEqual(
            contract.kv_bytes_per_token_per_device,
            2 * 32 * 8 * 128 * 2,
        )
        blocks = contract.kv_capacity_blocks_per_device
        self.assertIsNotNone(blocks)
        assert blocks is not None
        self.assertGreater(blocks, 0)
        self.assertEqual(contract.max_context_tokens, 8192)
        self.assertEqual(contract.max_sequences_at(2048), 128)
        self.assertEqual(contract.max_sequences_at(8192), 55)
        self.assertNotIn("runtime overhead reserve is zero", " ".join(contract.warnings))

    def test_exact_runtime_memory_budget_overrides_analytical_rounding(self) -> None:
        hardware = HardwareSpec("small-gpu", "nvidia", 1, 10_001, memory_utilization_limit=0.5)
        exact_runtime = runtime(
            memory_budget_bytes_per_device_override=5_001,
            runtime_overhead_bytes_per_device=1,
            activation_reserve_bytes_per_device=1,
        )
        model = model_7b(parameter_count=1, num_layers=1, num_kv_heads=1, head_dim=1, max_model_len=16)

        contract = capacity_for(model, hardware, exact_runtime)

        self.assertEqual(contract.memory_bytes_per_device["usable_budget"], 5_001)
        self.assertEqual(CapacityContract.from_dict(contract.to_dict()), contract)
        with self.assertRaisesRegex(ContractError, "does not match device memory"):
            capacity_for(
                model,
                hardware,
                replace(exact_runtime, memory_budget_bytes_per_device_override=10_002),
            )

    def test_concurrency_envelope_never_exceeds_kv_token_budget(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        previous = None
        capacity_blocks = contract.kv_capacity_blocks_per_device
        assert capacity_blocks is not None
        for point in contract.concurrency_envelope:
            used_blocks = point.blocks_per_sequence * point.max_sequences
            self.assertLessEqual(used_blocks, capacity_blocks)
            if previous is not None:
                self.assertLessEqual(point.max_sequences, previous)
            previous = point.max_sequences

    def test_explicit_weight_bytes_allow_quantized_or_sharded_artifacts(self) -> None:
        contract = capacity_for(
            model_7b(weight_dtype="unknown", explicit_weight_bytes=10 * GIB),
            HardwareSpec("mi300x", "amd", 1, 192 * GIB),
            runtime(engine="sglang", supported_vendors=("amd",)),
        )
        self.assertTrue(contract.fits)
        self.assertEqual(contract.model.weight_bytes, 10 * GIB)

    def test_incompatible_vendor_and_topology_never_claim_fit(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("a10", "nvidia", 1, 24 * GIB),
            runtime(
                tensor_parallel_size=2,
                kv_heads_per_device=4,
                supported_vendors=("amd",),
            ),
        )
        self.assertFalse(contract.fits)
        self.assertEqual(contract.max_sequences_at(1024), 0)
        self.assertGreaterEqual(len(contract.warnings), 2)

    def test_tensor_parallel_requires_explicit_kv_layout(self) -> None:
        with self.assertRaisesRegex(ContractError, "kv_heads_per_device"):
            capacity_for(
                model_7b(),
                HardwareSpec("h100x2", "nvidia", 2, 80 * GIB),
                runtime(tensor_parallel_size=2),
            )

    def test_oversized_weights_are_rejected(self) -> None:
        contract = capacity_for(
            model_7b(explicit_weight_bytes=100 * GIB),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        self.assertFalse(contract.fits)
        self.assertEqual(contract.kv_capacity_tokens_per_device, 0)

    def test_non_fitting_contract_ignores_requested_envelope_points(self) -> None:
        contract = capacity_for(
            model_7b(explicit_weight_bytes=100 * GIB),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
            context_points=(1024,),
        )
        self.assertFalse(contract.fits)
        self.assertEqual(contract.concurrency_envelope, ())

    def test_custom_attention_requires_context_bound_runtime_capacity(self) -> None:
        with self.assertRaises(ContractError):
            capacity_for(
                model_7b(attention_type="mla"),
                HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
                runtime(),
            )

        hardware = HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB)
        kv_memory = (80 * GIB * 9) // 10 - model_7b().weight_bytes - 3 * GIB
        exact_runtime = runtime(
            kv_capacity_hardware_id=hardware.hardware_id,
            kv_capacity_memory_bytes_per_device=kv_memory,
            kv_capacity_envelope_override=(
                RuntimeKVCapacityPoint(2048, 64),
                RuntimeKVCapacityPoint(8192, 12),
            ),
        )
        contract = capacity_for(
            model_7b(attention_type="hybrid"),
            hardware,
            exact_runtime,
            context_points=(8192,),
        )
        self.assertEqual(contract.kv_capacity_mode, KVCapacityMode.CONTEXT_ENVELOPE)
        self.assertIsNone(contract.kv_bytes_per_token_per_device)
        self.assertEqual(contract.max_sequences_at(8192), 12)
        self.assertEqual(contract.max_sequences_at(2048), 64)
        with self.assertRaisesRegex(ContractError, "no exact point"):
            capacity_for(
                model_7b(attention_type="hybrid"),
                hardware,
                exact_runtime,
                context_points=(4096,),
            )
        wrong_hardware = capacity_for(
            model_7b(attention_type="hybrid"),
            replace(hardware, hardware_id="other-h100"),
            exact_runtime,
        )
        self.assertFalse(wrong_hardware.fits)
        self.assertIn("different hardware ID", " ".join(wrong_hardware.warnings))
        wrong_memory = capacity_for(
            model_7b(attention_type="hybrid"),
            hardware,
            replace(exact_runtime, kv_capacity_memory_bytes_per_device=kv_memory + 1),
        )
        self.assertFalse(wrong_memory.fits)
        self.assertIn("different KV memory budget", " ".join(wrong_memory.warnings))
        with self.assertRaisesRegex(ContractError, "either a linear KV bytes/token override"):
            capacity_for(
                model_7b(attention_type="hybrid", kv_bytes_per_token_per_device_override=1),
                hardware,
                exact_runtime,
            )

    def test_reverse_fit_is_deterministic_and_puts_fits_first(self) -> None:
        inventory = HardwareInventory(
            (
                HardwareSpec("a10", "nvidia", 1, 16 * GIB),
                HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            )
        )
        candidates = what_fits(model_7b(), inventory, runtime())
        self.assertEqual(
            [candidate.hardware_id for candidate in candidates],
            ["h100", "a10"],
        )
        self.assertTrue(candidates[0].fits)
        self.assertFalse(candidates[1].fits)

    def test_reverse_fit_returns_unsupported_candidates_as_data(self) -> None:
        inventory = HardwareInventory((HardwareSpec("h100x2", "nvidia", 2, 80 * GIB),))
        candidates = what_fits(
            model_7b(),
            inventory,
            runtime(tensor_parallel_size=2),
        )
        self.assertEqual(len(candidates), 1)
        self.assertFalse(candidates[0].fits)
        self.assertIsNone(candidates[0].contract)
        self.assertIn("kv_heads_per_device", candidates[0].unsupported_reason or "")

    def test_round_trip_and_evidence_attachment(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        evidence = EvidenceRecord(
            evidence_id="run-001",
            kind=EvidenceKind.MEASURED,
            source="benchmark://run-001",
            scope="example/7b@sha256:example on h100/vllm-0.8.5",
            metrics={"prefill_tps": 1200},
        )
        restored = CapacityContract.from_dict(json.loads(json.dumps(contract.with_evidence(evidence).to_dict())))
        self.assertEqual(restored.evidence[0].kind, EvidenceKind.ANALYTICAL)
        self.assertEqual(restored.evidence[-1].evidence_id, "run-001")
        self.assertEqual(
            restored.validation_level,
            ValidationLevel.ANALYTICALLY_FEASIBLE,
        )
        self.assertEqual(restored.max_sequences_at(8192), 55)

    def test_contract_parser_rejects_coerced_boolean_and_inconsistent_kv_capacity(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        ).to_dict()
        contract["fits"] = "false"
        with self.assertRaisesRegex(ContractError, "boolean"):
            CapacityContract.from_dict(contract)

        contract["fits"] = True
        contract["kv_capacity_tokens_per_device"] = 1
        with self.assertRaisesRegex(ContractError, "block capacity"):
            CapacityContract.from_dict(contract)

    def test_adapters_do_not_invent_replica_count(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        planner = to_llmd_planner_payload(contract)
        scaling = to_scaling_policy_input(contract)
        self.assertEqual(planner["schema_version"], "llmd-capacity-input-3.0")
        self.assertIsNone(scaling["replica_count"])
        self.assertTrue(scaling["requires_observed_workload_profile"])
        self.assertIn("concurrency_envelope", scaling["per_replica_capacity"])

    def test_measured_profile_drives_scaling_cost_and_gpu_hours(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec(
                "h100",
                "nvidia",
                1,
                80 * GIB,
                price_per_device_hour=2.0,
            ),
            runtime(),
        )
        profile = WorkloadProfile(
            profile_id="profile-001",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=20,
            input_tokens_per_request=100,
            output_tokens_per_request=50,
            sustainable_requests_per_replica_per_second=10,
            sustainable_prefill_tokens_per_replica_per_second=1000,
            sustainable_decode_tokens_per_replica_per_second=800,
            target_utilization=0.5,
            scale_up_buffer=1.15,
            min_replicas=1,
            baseline_replicas=8,
            evidence=(
                EvidenceRecord(
                    evidence_id="run-002",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://run-002",
                    scope="example/7b@sha256:example on h100/vllm-0.8.5",
                    metrics={"sustainable_rps": 10},
                ),
            ),
        )
        recommendation = recommend_scale(contract, profile)
        self.assertEqual(recommendation.schema_version, "scaling-recommendation-2.0")
        self.assertEqual(recommendation.required_replicas, 4)
        self.assertEqual(recommendation.recommended_replicas, 5)
        self.assertEqual(recommendation.constrained_by, "requests_per_second")
        self.assertEqual(recommendation.estimated_gpu_hours_per_hour, 5.0)
        self.assertEqual(recommendation.estimated_gpu_hour_savings_vs_baseline, 3.0)
        self.assertEqual(recommendation.estimated_hourly_cost, 10.0)
        self.assertEqual(recommendation.estimated_hourly_savings_vs_baseline, 6.0)
        self.assertTrue(recommendation.within_bounds)

    def test_concurrency_driver_uses_context_specific_capacity(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(max_num_seqs=None),
        )
        profile = WorkloadProfile(
            profile_id="profile-concurrency",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=0,
            sustainable_requests_per_replica_per_second=1,
            sustainable_concurrent_sequences_per_replica=80,
            peak_concurrent_sequences=100,
            concurrency_context_tokens=8192,
            target_utilization=1.0,
            scale_up_buffer=1.0,
            evidence=(
                EvidenceRecord(
                    evidence_id="run-concurrency",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://run-concurrency",
                    scope="example/7b@sha256:example on h100/vllm-0.8.5",
                ),
            ),
        )
        recommendation = recommend_scale(contract, profile)
        self.assertEqual(contract.max_sequences_at(8192), 55)
        self.assertEqual(recommendation.required_replicas, 2)
        self.assertEqual(recommendation.constrained_by, "peak_concurrent_sequences")

    def test_concurrency_driver_clamps_measurement_to_analytical_kv_bound(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(max_num_seqs=None),
        )
        profile = WorkloadProfile(
            profile_id="profile-concurrency-clamp",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=0,
            sustainable_requests_per_replica_per_second=1,
            sustainable_concurrent_sequences_per_replica=100,
            peak_concurrent_sequences=100,
            concurrency_context_tokens=8192,
            target_utilization=1.0,
            scale_up_buffer=1.0,
            evidence=(
                EvidenceRecord(
                    evidence_id="run-concurrency-clamp",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://run-concurrency-clamp",
                    scope="example/7b@sha256:example on h100/vllm-0.8.5",
                ),
            ),
        )
        recommendation = recommend_scale(contract, profile)
        self.assertEqual(recommendation.required_replicas, 2)
        self.assertIn("analytical KV bound", " ".join(recommendation.warnings))

    def test_scaling_requires_measured_evidence_and_exact_scope(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        profile = WorkloadProfile(
            profile_id="profile-003",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=1,
            sustainable_requests_per_replica_per_second=1,
        )
        with self.assertRaises(ContractError):
            recommend_scale(contract, profile)
        reported = EvidenceRecord(
            evidence_id="report-001",
            kind=EvidenceKind.REPORTED,
            source="report://example",
            scope="example/7b",
        )
        reported_profile = replace(profile, evidence=(reported,))
        with self.assertRaises(ContractError):
            recommend_scale(contract, reported_profile)

    def test_scaling_marks_unmeasured_nonzero_drivers_incomplete(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        profile = WorkloadProfile(
            profile_id="profile-004",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=5,
            sustainable_prefill_tokens_per_replica_per_second=1000,
            evidence=(
                EvidenceRecord(
                    evidence_id="run-004",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://run-004",
                    scope="example/7b@sha256:example on h100/vllm-0.8.5",
                ),
            ),
        )
        recommendation = recommend_scale(contract, profile)
        self.assertFalse(recommendation.within_bounds)
        self.assertIn(
            "requests_per_second demand has no measured",
            " ".join(recommendation.warnings),
        )


if __name__ == "__main__":
    unittest.main()
