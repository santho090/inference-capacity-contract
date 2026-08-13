from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace

from inference_capacity_contract import (
    CapacityContract,
    ConcurrencyPoint,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareInventory,
    HardwareSpec,
    ModelSpec,
    RuntimeKVCapacityPoint,
    RuntimeVariant,
    WorkloadProfile,
    capacity_for,
    recommend_scale,
    what_fits,
)

GIB = 1024**3


def _model() -> ModelSpec:
    return ModelSpec("model", "revision", 100, 2, 2, 4, max_model_len=128)


def _hardware() -> HardwareSpec:
    return HardwareSpec("gpu", "nvidia", 1, GIB)


def _runtime() -> RuntimeVariant:
    return RuntimeVariant("vllm", "version", runtime_overhead_bytes_per_device=1)


def _evidence() -> EvidenceRecord:
    return EvidenceRecord("run", EvidenceKind.MEASURED, "benchmark://run", "exact variant")


def _profile(**overrides: object) -> WorkloadProfile:
    values: dict[str, object] = {
        "profile_id": "profile",
        "model_id": "model",
        "model_revision": "revision",
        "hardware_id": "gpu",
        "runtime_engine": "vllm",
        "runtime_version": "version",
        "request_rate_per_second": 1.0,
        "sustainable_requests_per_replica_per_second": 1.0,
        "evidence": (_evidence(),),
    }
    values.update(overrides)
    return WorkloadProfile(**values)  # type: ignore[arg-type]


class InputValidationTests(unittest.TestCase):
    def test_json_parsers_reject_numeric_strings_and_boolean_integers(self) -> None:
        model = _model().to_dict()
        model["parameter_count"] = "100"
        with self.assertRaisesRegex(ContractError, "integer"):
            ModelSpec.from_dict(model)
        missing_revision = _model().to_dict()
        del missing_revision["revision"]
        with self.assertRaisesRegex(ContractError, "missing required field 'revision'"):
            ModelSpec.from_dict(missing_revision)

        hardware = _hardware().to_dict()
        hardware["memory_bytes_per_device"] = "100"
        with self.assertRaisesRegex(ContractError, "integer"):
            HardwareSpec.from_dict(hardware)

        runtime = _runtime().to_dict()
        runtime["block_size_tokens"] = True
        with self.assertRaisesRegex(ContractError, "integer"):
            RuntimeVariant.from_dict(runtime)

        profile = _profile().to_dict()
        profile["request_rate_per_second"] = "1"
        with self.assertRaisesRegex(ContractError, "number"):
            WorkloadProfile.from_dict(profile)

        contract = capacity_for(_model(), _hardware(), _runtime()).to_dict()
        contract["memory_bytes_per_device"]["weights"] = "200"
        with self.assertRaisesRegex(ContractError, "integer"):
            CapacityContract.from_dict(contract)

        malformed_nested = capacity_for(_model(), _hardware(), _runtime()).to_dict()
        malformed_nested["runtime"]["supported_vendors"] = "nvidia"
        with self.assertRaisesRegex(ContractError, "array"):
            CapacityContract.from_dict(malformed_nested)

    def test_non_finite_numbers_are_rejected_at_the_domain_boundary(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ContractError):
                    HardwareSpec("gpu", "nvidia", 1, GIB, price_per_device_hour=value)
                with self.assertRaises(ContractError):
                    _profile(request_rate_per_second=value)
                with self.assertRaises(ContractError):
                    _profile(scale_up_buffer=value)

    def test_model_hardware_and_runtime_rules_reject_invalid_values(self) -> None:
        invalid_model_changes: tuple[dict[str, object], ...] = (
            {"model_id": ""},
            {"parameter_count": 0},
            {"weight_dtype": "unknown", "explicit_weight_bytes": None},
            {"attention_type": "unknown"},
            {"max_model_len": 0},
            {"kv_bytes_per_token_per_device_override": 0},
        )
        for changes in invalid_model_changes:
            with self.subTest(model=changes), self.assertRaises(ContractError):
                replace(_model(), **changes)  # type: ignore[arg-type]

        invalid_hardware_changes: tuple[dict[str, object], ...] = (
            {"vendor": ""},
            {"device_count": 0},
            {"memory_bytes_per_device": 0},
            {"memory_utilization_limit": 0},
            {"memory_utilization_limit": 1.1},
            {"price_per_device_hour": -1},
        )
        for changes in invalid_hardware_changes:
            with self.subTest(hardware=changes), self.assertRaises(ContractError):
                replace(_hardware(), **changes)  # type: ignore[arg-type]

        invalid_runtime_changes: tuple[dict[str, object], ...] = (
            {"engine": "unknown"},
            {"version": ""},
            {"tensor_parallel_size": 0},
            {"block_size_tokens": 0},
            {"runtime_overhead_bytes_per_device": -1},
            {"activation_reserve_bytes_per_device": -1},
            {"max_num_seqs": 0},
            {"kv_cache_dtype": "unknown"},
            {"supported_vendors": ()},
            {"weight_bytes_per_device_override": 0},
        )
        for changes in invalid_runtime_changes:
            with self.subTest(runtime=changes), self.assertRaises(ContractError):
                replace(_runtime(), **changes)  # type: ignore[arg-type]

        point = RuntimeKVCapacityPoint(1024, 8)
        with self.assertRaisesRegex(ContractError, "requires its hardware ID"):
            replace(_runtime(), kv_capacity_envelope_override=(point,))
        with self.assertRaisesRegex(ContractError, "require an envelope"):
            replace(_runtime(), kv_capacity_hardware_id="gpu", kv_capacity_memory_bytes_per_device=100)
        with self.assertRaisesRegex(ContractError, "strictly increasing"):
            replace(
                _runtime(),
                kv_capacity_hardware_id="gpu",
                kv_capacity_memory_bytes_per_device=100,
                kv_capacity_envelope_override=(point, point),
            )
        with self.assertRaisesRegex(ContractError, "cannot increase"):
            replace(
                _runtime(),
                kv_capacity_hardware_id="gpu",
                kv_capacity_memory_bytes_per_device=100,
                kv_capacity_envelope_override=(point, RuntimeKVCapacityPoint(2048, 9)),
            )
        with self.assertRaisesRegex(ContractError, "cannot exceed max_num_seqs"):
            replace(
                _runtime(),
                max_num_seqs=4,
                kv_capacity_hardware_id="gpu",
                kv_capacity_memory_bytes_per_device=100,
                kv_capacity_envelope_override=(point,),
            )
        with self.assertRaises(ContractError):
            RuntimeKVCapacityPoint(1024, 0)

        runtime_data = _runtime().to_dict()
        runtime_data["data_parallel_size"] = 2
        with self.assertRaisesRegex(ContractError, "data_parallel_size"):
            RuntimeVariant.from_dict(runtime_data)

    def test_evidence_profile_inventory_and_concurrency_rules_reject_invalid_values(self) -> None:
        with self.assertRaises(ContractError):
            EvidenceRecord("", EvidenceKind.MEASURED, "source", "scope")
        with self.assertRaises(ContractError):
            EvidenceRecord("id", "invalid", "source", "scope")  # type: ignore[arg-type]
        evidence_data = _evidence().to_dict()
        evidence_data["metrics"] = []
        with self.assertRaisesRegex(ContractError, "object"):
            EvidenceRecord.from_dict(evidence_data)

        invalid_profiles: tuple[dict[str, object], ...] = (
            {"profile_id": ""},
            {"request_rate_per_second": -1},
            {"sustainable_requests_per_replica_per_second": 0},
            {"target_utilization": 0},
            {"scale_up_buffer": 0.9},
            {"min_replicas": 0},
            {"max_replicas": 1, "min_replicas": 2},
            {"baseline_replicas": 0},
        )
        for changes in invalid_profiles:
            with self.subTest(profile=changes), self.assertRaises(ContractError):
                _profile(**changes)
        with self.assertRaisesRegex(ContractError, "at least one"):
            _profile(sustainable_requests_per_replica_per_second=None)
        with self.assertRaisesRegex(ContractError, "concurrency_context_tokens"):
            _profile(peak_concurrent_sequences=1)
        with self.assertRaisesRegex(ContractError, "sustainable_concurrent"):
            _profile(peak_concurrent_sequences=1, concurrency_context_tokens=1)

        with self.assertRaises(ContractError):
            HardwareInventory(())
        with self.assertRaises(ContractError):
            HardwareInventory((_hardware(), _hardware()))
        with self.assertRaises(ContractError):
            ConcurrencyPoint(1, 1, -1)
        with self.assertRaisesRegex(ContractError, "at least one runtime"):
            what_fits(_model(), HardwareInventory((_hardware(),)), ())

        candidate = what_fits(_model(), HardwareInventory((_hardware(),)), _runtime())[0]
        self.assertEqual(candidate.to_dict()["hardware_id"], "gpu")

    def test_contract_semantic_invariants_reject_tampering(self) -> None:
        original = capacity_for(_model(), _hardware(), _runtime()).to_dict()
        tampered_ledger = json.loads(json.dumps(original))
        tampered_ledger["memory_bytes_per_device"]["usable_budget"] = 0
        with self.assertRaisesRegex(ContractError, "memory_bytes_per_device is inconsistent"):
            CapacityContract.from_dict(tampered_ledger)

        inconsistent_fit = {**original, "fits": False}
        with self.assertRaisesRegex(ContractError, "fits is inconsistent"):
            CapacityContract.from_dict(inconsistent_fit)

        mutations = (
            ("schema_version", "wrong"),
            ("kv_bytes_per_token_per_device", 0),
            ("kv_capacity_blocks_per_device", -1),
            ("max_context_tokens", original["kv_capacity_tokens_per_device"] + 1),
        )
        for field, value in mutations:
            document = {**original, field: value}
            with self.subTest(field=field), self.assertRaises((ContractError, ValueError)):
                CapacityContract.from_dict(document)

        missing_memory = {**original, "memory_bytes_per_device": {"weights": 1}}
        with self.assertRaisesRegex(ContractError, "complete"):
            CapacityContract.from_dict(missing_memory)

        inconsistent_tokens = {**original, "kv_capacity_tokens_per_device": 1}
        with self.assertRaisesRegex(ContractError, "block capacity"):
            CapacityContract.from_dict(inconsistent_tokens)

        no_envelope = {**original, "concurrency_envelope": []}
        with self.assertRaisesRegex(ContractError, "concurrency envelope"):
            CapacityContract.from_dict(no_envelope)

        no_evidence = {**original, "evidence": []}
        with self.assertRaisesRegex(ContractError, "provenance"):
            CapacityContract.from_dict(no_evidence)


class ScalingRuleTests(unittest.TestCase):
    def test_identity_dimensions_are_checked_independently(self) -> None:
        contract = capacity_for(_model(), _hardware(), _runtime())
        mismatches = (
            ({"model_id": "other"}, "model identity"),
            ({"model_revision": "other"}, "model identity"),
            ({"hardware_id": "other"}, "hardware identity"),
            ({"runtime_engine": "sglang"}, "runtime identity"),
            ({"runtime_version": "other"}, "runtime identity"),
        )
        for changes, message in mismatches:
            with self.subTest(changes=changes), self.assertRaisesRegex(ContractError, message):
                recommend_scale(contract, _profile(**changes))

    def test_bounds_zero_demand_and_scale_up_suggestions_are_consistent(self) -> None:
        contract = capacity_for(_model(), replace(_hardware(), price_per_device_hour=2), _runtime())
        capped = recommend_scale(
            contract,
            _profile(
                request_rate_per_second=10,
                sustainable_requests_per_replica_per_second=1,
                target_utilization=1,
                scale_up_buffer=1,
                max_replicas=4,
                baseline_replicas=2,
            ),
        )
        self.assertEqual(capped.required_replicas, 10)
        self.assertEqual(capped.recommended_replicas, 4)
        self.assertFalse(capped.within_bounds)
        self.assertLess(capped.estimated_gpu_hour_savings_vs_baseline or 0, 0)
        self.assertIn("increasing replicas", " ".join(capped.suggestions))

        idle = recommend_scale(
            contract,
            _profile(request_rate_per_second=0, min_replicas=2, baseline_replicas=4),
        )
        self.assertEqual(idle.required_replicas, 0)
        self.assertEqual(idle.recommended_replicas, 2)
        self.assertTrue(idle.within_bounds)
        self.assertEqual(idle.constrained_by, "min-replicas-floor")
        self.assertEqual(idle.drivers, {})
        self.assertIn("reducing replicas", " ".join(idle.suggestions))

    def test_unused_concurrency_measurement_does_not_emit_a_false_bound_warning(self) -> None:
        contract = capacity_for(_model(), _hardware(), _runtime())
        recommendation = recommend_scale(
            contract,
            _profile(
                request_rate_per_second=0,
                sustainable_concurrent_sequences_per_replica=1_000,
                concurrency_context_tokens=128,
            ),
        )
        self.assertNotIn("analytical KV bound", " ".join(recommendation.warnings))


if __name__ == "__main__":
    unittest.main()
