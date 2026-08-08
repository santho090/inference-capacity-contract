import json
import unittest

from inference_capacity_contract import (
    CapacityContract,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareInventory,
    HardwareSpec,
    ModelSpec,
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


def model_7b(**overrides: object) -> ModelSpec:
    values: dict[str, object] = {
        "model_id": "example/7b",
        "revision": "sha256:example",
        "parameter_count": 7_000_000_000,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "weight_dtype": "bf16",
        "max_model_len": 8192,
    }
    values.update(overrides)
    return ModelSpec(**values)


def runtime(**overrides: object) -> RuntimeVariant:
    values: dict[str, object] = {
        "engine": "vllm",
        "version": "0.8.5",
        "runtime_overhead_bytes_per_device": 1 * GIB,
        "activation_reserve_bytes_per_device": 2 * GIB,
        "max_num_seqs": 128,
    }
    values.update(overrides)
    return RuntimeVariant(**values)


class CalculatorTests(unittest.TestCase):
    def test_analytical_fit_accounts_for_weights_reserves_and_kv(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        self.assertTrue(contract.fits)
        self.assertEqual(contract.validation_level, ValidationLevel.ANALYTICALLY_FEASIBLE)
        self.assertEqual(contract.kv_bytes_per_token, 2 * 32 * 8 * 128 * 2)
        self.assertGreater(contract.kv_capacity_tokens, 0)
        self.assertLessEqual(contract.max_context_tokens, 8192)
        self.assertLessEqual(contract.max_concurrent_sequences, 128)
        self.assertNotIn("runtime overhead reserve is zero", " ".join(contract.warnings))

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
            runtime(tensor_parallel_size=2, supported_vendors=("amd",)),
        )
        self.assertFalse(contract.fits)
        self.assertEqual(contract.max_concurrent_sequences, 0)
        self.assertGreaterEqual(len(contract.warnings), 2)

    def test_oversized_weights_are_rejected(self) -> None:
        contract = capacity_for(
            model_7b(explicit_weight_bytes=100 * GIB),
            HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        self.assertFalse(contract.fits)
        self.assertEqual(contract.kv_capacity_tokens, 0)

    def test_custom_attention_requires_explicit_kv_formula(self) -> None:
        with self.assertRaises(ContractError):
            capacity_for(
                model_7b(attention_type="mla"),
                HardwareSpec("h100-80gb", "nvidia", 1, 80 * GIB),
                runtime(),
            )

    def test_reverse_fit_is_deterministic_and_puts_fits_first(self) -> None:
        inventory = HardwareInventory(
            (
                HardwareSpec("a10", "nvidia", 1, 16 * GIB),
                HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            )
        )
        candidates = what_fits(model_7b(), inventory, runtime())
        self.assertEqual([candidate.hardware_id for candidate in candidates], ["h100", "a10"])
        self.assertTrue(candidates[0].fits)
        self.assertFalse(candidates[1].fits)

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
        enriched = contract.with_evidence(evidence)
        restored = CapacityContract.from_dict(json.loads(json.dumps(enriched.to_dict())))
        self.assertEqual(restored.evidence[0].evidence_id, "run-001")
        self.assertEqual(restored.validation_level, ValidationLevel.ANALYTICALLY_FEASIBLE)

    def test_adapters_do_not_invent_replica_count(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            runtime(),
        )
        planner = to_llmd_planner_payload(contract)
        scaling = to_scaling_policy_input(contract)
        self.assertEqual(planner["schema_version"], "llmd-capacity-input-1.0")
        self.assertIsNone(scaling["replica_count"])
        self.assertTrue(scaling["requires_observed_workload_profile"])

    def test_measured_profile_drives_transparent_scaling_and_cost_savings(self) -> None:
        contract = capacity_for(
            model_7b(),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB, price_per_device_hour=2.0),
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
        self.assertEqual(recommendation.required_replicas, 4)
        self.assertEqual(recommendation.recommended_replicas, 5)
        self.assertEqual(recommendation.constrained_by, "requests_per_second")
        self.assertEqual(recommendation.estimated_hourly_cost, 10.0)
        self.assertEqual(recommendation.estimated_hourly_savings_vs_baseline, 6.0)
        self.assertTrue(recommendation.within_bounds)

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
        reported_profile = WorkloadProfile(
            **{**profile.to_dict(), "evidence": (reported,)},
        )
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
        self.assertIn("requests_per_second demand has no measured", " ".join(recommendation.warnings))


if __name__ == "__main__":
    unittest.main()
