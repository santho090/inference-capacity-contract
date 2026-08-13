from __future__ import annotations

import json
import unittest
from dataclasses import fields, replace

from inference_capacity_contract import (
    RECIPE_VARIANT_FINGERPRINT_VERSION,
    AuditStatus,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    LLMDRoutingSpec,
    LoadRequirement,
    MeasuredGroupProfile,
    ModelSpec,
    ParallelTopology,
    RuntimeKVCapacityPoint,
    RuntimeVariant,
    ServingRecipe,
    audit_recipe,
)

GIB = 1024**3


def _evidence(
    kind: EvidenceKind = EvidenceKind.REPORTED,
    metrics: dict[str, float | int | str] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"{kind.value}-fixture",
        kind=kind,
        source="fixture://sanitized-serving-recipe",
        scope="sanitized model/runtime/host shape",
        metrics={} if metrics is None else metrics,
    )


def _model(**changes: object) -> ModelSpec:
    values: dict[str, object] = {
        "model_id": "example/custom-moe",
        "revision": "sha256:fixture",
        "parameter_count": 1,
        "num_layers": 1,
        "num_kv_heads": 1,
        "head_dim": 1,
        "weight_dtype": "unknown",
        "max_model_len": 1_048_576,
        "explicit_weight_bytes": 1_280 * GIB,
        "attention_type": "custom",
        "kv_bytes_per_token_per_device_override": None,
    }
    values.update(changes)
    return ModelSpec(**values)  # type: ignore[arg-type]


def _host(**changes: object) -> HardwareSpec:
    values: dict[str, object] = {
        "hardware_id": "example-accelerator-x8",
        "vendor": "nvidia",
        "device_count": 8,
        "memory_bytes_per_device": 288 * GIB,
        "memory_utilization_limit": 0.96,
    }
    values.update(changes)
    return HardwareSpec(**values)  # type: ignore[arg-type]


def _runtime(**changes: object) -> RuntimeVariant:
    values: dict[str, object] = {
        "engine": "vllm",
        "version": "fixture",
        "tensor_parallel_size": 8,
        "kv_cache_dtype": "fp8",
        "weight_bytes_per_device_override": 160 * GIB,
        "runtime_overhead_bytes_per_device": 8 * GIB,
        "activation_reserve_bytes_per_device": 8 * GIB,
        "max_num_seqs": 24,
        "block_size_tokens": 256,
    }
    values.update(changes)
    if "kv_capacity_hardware_id" not in changes:
        values["kv_capacity_hardware_id"] = "example-accelerator-x8"
    if "kv_capacity_memory_bytes_per_device" not in changes:
        usable = (288 * GIB * 96) // 100
        weight_value = values["weight_bytes_per_device_override"]
        weights = weight_value if isinstance(weight_value, int) else 0
        runtime_value = values["runtime_overhead_bytes_per_device"]
        activation_value = values["activation_reserve_bytes_per_device"]
        assert isinstance(runtime_value, int) and isinstance(activation_value, int)
        runtime_reserve = runtime_value
        activation_reserve = activation_value
        values["kv_capacity_memory_bytes_per_device"] = max(
            1,
            usable - weights - runtime_reserve - activation_reserve,
        )
    if "kv_capacity_envelope_override" not in changes:
        values["kv_capacity_envelope_override"] = (
            RuntimeKVCapacityPoint(1024, 24),
            RuntimeKVCapacityPoint(65_536, 24),
            RuntimeKVCapacityPoint(262_144, 24),
            RuntimeKVCapacityPoint(1_048_576, 24),
        )
    return RuntimeVariant(**values)  # type: ignore[arg-type]


def _tp_recipe(**changes: object) -> ServingRecipe:
    values: dict[str, object] = {
        "schema_version": "serving-recipe-1.0",
        "recipe_id": "sanitized-hybrid-tp8",
        "model": _model(),
        "host": _host(),
        "runtime": _runtime(),
        "topology": ParallelTopology(
            tensor_parallel_size=8,
            data_parallel_size=1,
            expert_parallel_size=8,
            physical_device_count=8,
            independent_kv_ranks=1,
        ),
        "llmd": LLMDRoutingSpec(
            flow_control_token_limit=100_000_000,
            max_concurrent_sequences=24,
            kv_block_size_tokens=256,
            output_ratio=0.2,
            max_estimated_output_tokens=500,
            precise_prefix_routing=True,
        ),
        "configured_groups": 2,
        "evidence": (_evidence(),),
    }
    values.update(changes)
    return ServingRecipe(**values)  # type: ignore[arg-type]


def _load(**changes: object) -> LoadRequirement:
    values: dict[str, object] = {
        "schema_version": "load-requirement-1.0",
        "context_tokens": 1_048_576,
        "peak_concurrent_sequences": 30,
        "target_utilization": 0.8,
    }
    values.update(changes)
    return LoadRequirement(**values)  # type: ignore[arg-type]


def _profile(recipe: ServingRecipe, **changes: object) -> MeasuredGroupProfile:
    values: dict[str, object] = {
        "profile_id": "sanitized-measured-group",
        "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
        "recipe_variant_fingerprint": recipe.variant_fingerprint,
        "context_tokens": 1_048_576,
        "input_tokens_per_request": 100,
        "output_tokens_per_request": 20,
        "requests_per_second": 10,
        "prefill_tokens_per_second": 1000,
        "decode_tokens_per_second": 200,
        "concurrent_sequences": 20,
        "ttft_ms": 150,
        "latency_percentile": 99,
    }
    values.update(changes)
    if "evidence" not in changes:
        metric_names = (
            "requests_per_second",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "concurrent_sequences",
            "ttft_ms",
            "tpot_ms",
            "latency_percentile",
        )
        fingerprint = values["recipe_variant_fingerprint"]
        context = values["context_tokens"]
        input_tokens = values["input_tokens_per_request"]
        output_tokens = values["output_tokens_per_request"]
        assert isinstance(fingerprint, str)
        assert isinstance(context, int)
        assert isinstance(input_tokens, (int, float))
        assert isinstance(output_tokens, (int, float))
        metrics: dict[str, float | int | str] = {
            "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
            "recipe_variant_fingerprint": fingerprint,
            "context_tokens": context,
            "input_tokens_per_request": input_tokens,
            "output_tokens_per_request": output_tokens,
        }
        for name in metric_names:
            metric = values.get(name)
            if metric is not None:
                assert isinstance(metric, (int, float))
                metrics[name] = metric
        values["evidence"] = (_evidence(EvidenceKind.MEASURED, metrics),)
    return MeasuredGroupProfile(**values)  # type: ignore[arg-type]


class RecipeAuditTests(unittest.TestCase):
    def test_tp8_recipe_reports_required_groups_at_requested_context(self) -> None:
        result = audit_recipe(_tp_recipe(), _load())
        self.assertEqual(result.status, AuditStatus.SUFFICIENT)
        self.assertTrue(result.context_supported)
        self.assertEqual(result.analytical_sequences_per_group, 24)
        self.assertEqual(result.effective_sequences_per_group, 24)
        self.assertEqual(result.drivers["peak_concurrent_sequences"], 2)
        self.assertEqual(result.required_groups, 2)
        self.assertEqual(result.required_devices, 16)
        self.assertEqual(result.configured_devices, 16)
        self.assertEqual(result.additional_groups_needed, 0)
        self.assertEqual(result.additional_devices_needed, 0)
        self.assertTrue(result.configured_load_satisfied)

    def test_dp_ep_recipe_aggregates_independent_kv_ranks(self) -> None:
        topology = ParallelTopology(
            tensor_parallel_size=1,
            data_parallel_size=8,
            expert_parallel_size=8,
            physical_device_count=8,
            independent_kv_ranks=8,
        )
        recipe = _tp_recipe(
            recipe_id="sanitized-sparse-dp8-ep8",
            model=_model(max_model_len=262_144),
            runtime=_runtime(
                tensor_parallel_size=1,
                weight_bytes_per_device_override=200 * GIB,
                max_num_seqs=128,
                kv_capacity_envelope_override=(RuntimeKVCapacityPoint(262_144, 120),),
            ),
            topology=topology,
            llmd=LLMDRoutingSpec(
                flow_control_token_limit=240_000_000,
                max_concurrent_sequences=1024,
                kv_block_size_tokens=256,
                precise_prefix_routing=True,
            ),
            configured_groups=1,
        )
        result = audit_recipe(
            recipe,
            _load(context_tokens=262_144, peak_concurrent_sequences=700),
        )
        self.assertEqual(result.status, AuditStatus.SUFFICIENT)
        self.assertEqual(result.analytical_sequences_per_kv_rank, 120)
        self.assertEqual(result.analytical_sequences_per_group, 960)
        self.assertEqual(result.effective_sequences_per_group, 915)
        self.assertEqual(result.calculated_max_concurrent_sequences, 1024)
        self.assertEqual(result.required_groups, 1)

    def test_flow_control_clamps_full_context_capacity(self) -> None:
        recipe = _tp_recipe(
            llmd=LLMDRoutingSpec(
                flow_control_token_limit=2_104_030,
                max_concurrent_sequences=24,
                kv_block_size_tokens=256,
                precise_prefix_routing=True,
            ),
            configured_groups=1,
        )

        result = audit_recipe(recipe, _load())

        self.assertEqual(result.status, AuditStatus.INSUFFICIENT)
        self.assertEqual(result.analytical_sequences_per_group, 24)
        self.assertEqual(
            result.sequence_capacity_limits,
            {
                "runtime_kv_envelope": 24,
                "runtime": 24,
                "llmd_flow_control": 2,
                "llmd_max_concurrency": 24,
            },
        )
        self.assertEqual(result.effective_sequences_per_group, 2)
        self.assertEqual(result.drivers["peak_concurrent_sequences"], 19)
        self.assertEqual(result.required_groups, 19)
        self.assertEqual(result.required_devices, 152)
        self.assertEqual(result.additional_groups_needed, 18)
        self.assertEqual(result.additional_devices_needed, 144)
        self.assertTrue(any("not divisible" in warning for warning in result.warnings))

    def test_zero_flow_capacity_has_no_finite_group_count(self) -> None:
        recipe = _tp_recipe(
            llmd=LLMDRoutingSpec(
                flow_control_token_limit=1_000_000,
                max_concurrent_sequences=24,
                kv_block_size_tokens=256,
                precise_prefix_routing=True,
            )
        )

        result = audit_recipe(recipe, _load())

        self.assertEqual(result.status, AuditStatus.INSUFFICIENT)
        self.assertEqual(result.effective_sequences_per_group, 0)
        self.assertIsNone(result.required_groups)
        self.assertIsNone(result.required_devices)
        self.assertIsNone(result.additional_groups_needed)
        self.assertIsNone(result.additional_devices_needed)

        traffic_result = audit_recipe(
            recipe,
            _load(
                peak_concurrent_sequences=0,
                request_rate_per_second=1,
                input_tokens_per_request=100,
                measured_profile=_profile(recipe),
            ),
        )
        self.assertEqual(traffic_result.status, AuditStatus.INSUFFICIENT)
        self.assertIsNone(traffic_result.required_groups)

    def test_recipe_rejects_invalid_topology_and_missing_ep_weight_layout(self) -> None:
        with self.assertRaisesRegex(ContractError, "TP x DP"):
            ParallelTopology(tensor_parallel_size=2, data_parallel_size=2, physical_device_count=3)
        with self.assertRaisesRegex(ContractError, "weight_bytes_per_device_override"):
            _tp_recipe(runtime=replace(_runtime(), weight_bytes_per_device_override=None))

    def test_audit_finds_routing_mismatches_and_unused_devices(self) -> None:
        topology = ParallelTopology(
            tensor_parallel_size=1,
            data_parallel_size=4,
            expert_parallel_size=4,
            physical_device_count=8,
            independent_kv_ranks=4,
        )
        recipe = _tp_recipe(
            runtime=_runtime(tensor_parallel_size=1, max_num_seqs=128),
            topology=topology,
            llmd=LLMDRoutingSpec(
                flow_control_token_limit=10**12,
                max_concurrent_sequences=128,
                kv_block_size_tokens=128,
                precise_prefix_routing=True,
            ),
        )
        result = audit_recipe(recipe, _load(context_tokens=1024, peak_concurrent_sequences=1))
        self.assertEqual(result.status, AuditStatus.INVALID)
        self.assertIn("llm-d and vLLM block sizes do not match", result.issues)
        self.assertIsNone(result.calculated_kv_tokens_per_group)
        self.assertIn("llm-d max concurrency", " ".join(result.issues))
        self.assertIn("4 physical devices", " ".join(result.warnings))

    def test_nonzero_traffic_without_measurements_is_incomplete(self) -> None:
        result = audit_recipe(
            _tp_recipe(),
            _load(request_rate_per_second=2, input_tokens_per_request=100, output_tokens_per_request=20),
        )
        self.assertEqual(result.status, AuditStatus.INCOMPLETE)
        self.assertIsNone(result.required_groups)
        self.assertIsNone(result.additional_groups_needed)
        self.assertIsNone(result.additional_devices_needed)
        self.assertIsNone(result.configured_load_satisfied)
        self.assertIn("requests_per_second", " ".join(result.warnings))

    def test_measured_traffic_and_latency_can_pass_or_fail(self) -> None:
        recipe = _tp_recipe()
        passing = audit_recipe(
            recipe,
            _load(
                request_rate_per_second=10,
                input_tokens_per_request=100,
                output_tokens_per_request=20,
                target_ttft_ms=200,
                target_latency_percentile=99,
                measured_profile=_profile(recipe),
            ),
        )
        self.assertEqual(passing.status, AuditStatus.SUFFICIENT)
        self.assertEqual(passing.required_groups, 2)

        failing = audit_recipe(
            recipe,
            replace(
                passing.load,
                measured_profile=_profile(recipe, ttft_ms=250),
            ),
        )
        self.assertEqual(failing.status, AuditStatus.INSUFFICIENT)
        self.assertIn("observed TTFT", " ".join(failing.issues))

    def test_group_shortfall_is_reported_directly(self) -> None:
        result = audit_recipe(_tp_recipe(configured_groups=1), _load())
        self.assertEqual(result.status, AuditStatus.INSUFFICIENT)
        self.assertEqual(result.required_groups, 2)
        self.assertEqual(result.configured_devices, 8)
        self.assertEqual(result.additional_groups_needed, 1)
        self.assertEqual(result.additional_devices_needed, 8)

    def test_context_and_memory_failures_are_invalid_configurations(self) -> None:
        context_failure = audit_recipe(
            _tp_recipe(model=_model(max_model_len=32_768)),
            _load(context_tokens=65_536, peak_concurrent_sequences=1),
        )
        self.assertEqual(context_failure.status, AuditStatus.INSUFFICIENT)
        self.assertFalse(context_failure.context_supported)

        memory_failure = audit_recipe(
            _tp_recipe(runtime=_runtime(weight_bytes_per_device_override=300 * GIB)),
            _load(context_tokens=1024, peak_concurrent_sequences=1),
        )
        self.assertEqual(memory_failure.status, AuditStatus.INVALID)
        self.assertIn("do not fit", " ".join(memory_failure.issues))

        context_and_missing_measurement = audit_recipe(
            _tp_recipe(),
            _load(
                context_tokens=2_000_000,
                peak_concurrent_sequences=1,
                request_rate_per_second=1,
            ),
        )
        self.assertEqual(context_and_missing_measurement.status, AuditStatus.INSUFFICIENT)
        self.assertIsNone(context_and_missing_measurement.additional_devices_needed)

    def test_measured_concurrency_cannot_raise_the_analytical_limit(self) -> None:
        recipe = _tp_recipe(configured_groups=3)
        result = audit_recipe(
            recipe,
            _load(
                context_tokens=1_048_576,
                peak_concurrent_sequences=30,
                measured_profile=_profile(recipe, concurrent_sequences=100),
            ),
        )
        self.assertEqual(result.analytical_sequences_per_group, 24)
        self.assertEqual(result.required_groups, 2)

    def test_direct_tps_demand_works_without_rps(self) -> None:
        recipe = _tp_recipe(configured_groups=2)
        result = audit_recipe(
            recipe,
            _load(
                prefill_tokens_per_second=900,
                decode_tokens_per_second=180,
                measured_profile=_profile(recipe),
            ),
        )
        self.assertEqual(result.status, AuditStatus.SUFFICIENT)
        self.assertEqual(result.drivers["prefill_tokens_per_second"], 2)
        self.assertEqual(result.drivers["decode_tokens_per_second"], 2)
        with self.assertRaisesRegex(ContractError, "prefill TPS conflicts"):
            _load(
                request_rate_per_second=2,
                input_tokens_per_request=100,
                prefill_tokens_per_second=201,
            )

    def test_exact_utilization_boundary_does_not_add_a_group(self) -> None:
        recipe = _tp_recipe(configured_groups=1)
        profile = _profile(recipe)
        exact = audit_recipe(
            recipe,
            _load(peak_concurrent_sequences=0, request_rate_per_second=8, measured_profile=profile),
        )
        above = audit_recipe(
            recipe,
            _load(peak_concurrent_sequences=0, request_rate_per_second=8.0000001, measured_profile=profile),
        )
        self.assertEqual(exact.required_groups, 1)
        self.assertEqual(above.required_groups, 2)

    def test_measurement_profile_must_match_recipe_and_each_metric(self) -> None:
        recipe = _tp_recipe()
        other_recipe = replace(recipe, runtime=replace(recipe.runtime, block_size_tokens=512))
        mismatched = _profile(other_recipe)
        result = audit_recipe(
            recipe,
            _load(request_rate_per_second=1, measured_profile=mismatched),
        )
        self.assertEqual(result.status, AuditStatus.INVALID)
        self.assertIn("fingerprint", " ".join(result.issues))

        percentile_mismatch = audit_recipe(
            recipe,
            _load(
                request_rate_per_second=1,
                target_ttft_ms=200,
                target_latency_percentile=95,
                measured_profile=_profile(recipe),
            ),
        )
        self.assertEqual(percentile_mismatch.status, AuditStatus.INVALID)
        self.assertIn("percentile", " ".join(percentile_mismatch.issues))

        for bad_profile, expected_issue in (
            (_profile(recipe, context_tokens=8192), "context"),
            (_profile(recipe, input_tokens_per_request=101, prefill_tokens_per_second=1010), "input shape"),
        ):
            scoped = audit_recipe(
                recipe,
                _load(
                    request_rate_per_second=1,
                    input_tokens_per_request=100,
                    output_tokens_per_request=20,
                    measured_profile=bad_profile,
                ),
            )
            self.assertEqual(scoped.status, AuditStatus.INVALID)
            self.assertIn(expected_issue, " ".join(scoped.issues))

        with self.assertRaisesRegex(ContractError, "one measured evidence record"):
            _profile(
                recipe,
                requests_per_second=11,
                prefill_tokens_per_second=1100,
                decode_tokens_per_second=220,
                evidence=(_profile(recipe).evidence[0],),
            )

        with self.assertRaisesRegex(ContractError, "one measured evidence record"):
            _profile(
                recipe,
                evidence=(
                    _evidence(
                        EvidenceKind.MEASURED,
                        {
                            "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
                            "recipe_variant_fingerprint": recipe.variant_fingerprint,
                            "context_tokens": 1_048_576,
                            "input_tokens_per_request": 100,
                            "output_tokens_per_request": 20,
                        },
                    ),
                    _evidence(
                        EvidenceKind.MEASURED,
                        {
                            "requests_per_second": 10,
                            "prefill_tokens_per_second": 1000,
                            "decode_tokens_per_second": 200,
                            "concurrent_sequences": 20,
                            "ttft_ms": 150,
                            "latency_percentile": 99,
                        },
                    ),
                ),
            )

        with self.assertRaisesRegex(ContractError, "traffic operating point"):
            MeasuredGroupProfile(
                profile_id="latency-only",
                recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
                recipe_variant_fingerprint=recipe.variant_fingerprint,
                context_tokens=1024,
                ttft_ms=100,
                evidence=(
                    _evidence(
                        EvidenceKind.MEASURED,
                        {
                            "recipe_variant_fingerprint": recipe.variant_fingerprint,
                            "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
                            "context_tokens": 1024,
                            "input_tokens_per_request": 0,
                            "output_tokens_per_request": 0,
                            "ttft_ms": 100,
                            "latency_percentile": 99,
                        },
                    ),
                ),
                latency_percentile=99,
            )

    def test_recipe_fingerprint_tracks_performance_shape_not_count_or_price(self) -> None:
        recipe = _tp_recipe()
        self.assertEqual(recipe.variant_fingerprint, replace(recipe, configured_groups=9).variant_fingerprint)
        self.assertEqual(
            recipe.variant_fingerprint,
            replace(recipe, host=replace(recipe.host, price_per_device_hour=99)).variant_fingerprint,
        )
        numeric_a = replace(recipe, host=replace(recipe.host, memory_utilization_limit=1))
        numeric_b = replace(recipe, host=replace(recipe.host, memory_utilization_limit=1.0))
        self.assertEqual(numeric_a.variant_fingerprint, numeric_b.variant_fingerprint)
        metadata_a = replace(
            recipe,
            runtime=replace(recipe.runtime, supported_vendors=("nvidia", "amd"), notes=("first",)),
        )
        metadata_b = replace(
            recipe,
            runtime=replace(recipe.runtime, supported_vendors=("amd", "nvidia"), notes=("second",)),
        )
        self.assertEqual(metadata_a.variant_fingerprint, metadata_b.variant_fingerprint)
        self.assertNotEqual(
            recipe.variant_fingerprint,
            replace(recipe, runtime=replace(recipe.runtime, block_size_tokens=512)).variant_fingerprint,
        )
        self.assertEqual(
            recipe.variant_fingerprint,
            "sha256:574a4f7d0fb677b4d9b01b7433e3916369092d591367d3e1502dd0add43a8f8b",
        )
        included = set(recipe._variant_document()) - {"fingerprint_version"}
        excluded = {"schema_version", "recipe_id", "configured_groups", "evidence"}
        self.assertEqual({field.name for field in fields(ServingRecipe)}, included | excluded)
        document = recipe._variant_document()
        self.assertEqual(
            {field.name for field in fields(HardwareSpec)},
            set(document["host"]) | {"price_per_device_hour"},
        )
        self.assertEqual(
            {field.name for field in fields(RuntimeVariant)},
            set(document["runtime"]) | {"notes", "supported_vendors"},
        )

    def test_round_trip_is_strict_and_stable(self) -> None:
        recipe_document = json.loads(json.dumps(_tp_recipe().to_dict()))
        load_document = json.loads(json.dumps(_load().to_dict()))
        restored_recipe = ServingRecipe.from_dict(recipe_document)
        restored_load = LoadRequirement.from_dict(load_document)
        self.assertEqual(restored_recipe.to_dict(), recipe_document)
        self.assertEqual(restored_load.to_dict(), load_document)

        recipe_document["topology"]["physical_device_count"] = 7
        with self.assertRaises(ContractError):
            ServingRecipe.from_dict(recipe_document)
        with self.assertRaisesRegex(ContractError, "measured evidence"):
            MeasuredGroupProfile(
                profile_id="missing-evidence",
                recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
                recipe_variant_fingerprint=_tp_recipe().variant_fingerprint,
                context_tokens=1024,
                requests_per_second=1,
            )

        recipe_document = _tp_recipe().to_dict()
        recipe_document["schema_version"] = 1
        with self.assertRaisesRegex(ContractError, "non-empty string"):
            ServingRecipe.from_dict(recipe_document)

        llmd_document = _tp_recipe().llmd.to_dict()
        llmd_document["output_ratio"] = 0
        self.assertEqual(LLMDRoutingSpec.from_dict(llmd_document).output_ratio, 0)

        for invalid in (float("nan"), float("inf"), -1):
            with self.assertRaises(ContractError):
                _load(request_rate_per_second=invalid)
        with self.assertRaisesRegex(ContractError, "target_latency_percentile"):
            _load(target_ttft_ms=100)


if __name__ == "__main__":
    unittest.main()
