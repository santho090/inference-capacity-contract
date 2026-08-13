from __future__ import annotations

import unittest
from dataclasses import replace

from inference_capacity_contract import (
    RECIPE_VARIANT_FINGERPRINT_VERSION,
    AuditStatus,
    CandidateStatus,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    LLMDRoutingSpec,
    LoadRequirement,
    MeasuredGroupProfile,
    MeasurementInventory,
    ModelSpec,
    PlanningObjective,
    ProviderInstanceSpec,
    ProviderInventory,
    RuntimeInventory,
    RuntimeOption,
    RuntimeVariant,
    explore,
    import_huggingface_manifest,
    plan,
)

GIB = 1024**3


def _model(**changes: object) -> ModelSpec:
    values: dict[str, object] = {
        "model_id": "example/7b",
        "revision": "sha256:fixture",
        "parameter_count": 7_000_000_000,
        "num_layers": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "weight_dtype": "bf16",
        "max_model_len": 8192,
    }
    values.update(changes)
    return ModelSpec(**values)  # type: ignore[arg-type]


def _provider(**changes: object) -> ProviderInstanceSpec:
    values: dict[str, object] = {
        "provider_id": "provider-a",
        "region": "region-1",
        "instance_type": "accelerator-80gb-x1",
        "accelerator_id": "accelerator-80gb",
        "vendor": "nvidia",
        "devices_per_instance": 1,
        "memory_bytes_per_device": 80 * GIB,
        "memory_utilization_limit": 0.9,
        "price_per_instance_hour": 4.0,
        "price_currency": "USD",
        "max_instances": 10,
        "source": "inventory://provider-a",
        "collected_at": "2026-08-12T12:00:00Z",
    }
    values.update(changes)
    return ProviderInstanceSpec(**values)  # type: ignore[arg-type]


def _runtime(**changes: object) -> RuntimeOption:
    runtime_changes = changes.pop("runtime", {})
    assert isinstance(runtime_changes, dict)
    runtime_values: dict[str, object] = {
        "engine": "vllm",
        "version": "1.0.0",
        "tensor_parallel_size": 1,
        "kv_cache_dtype": "bf16",
        "runtime_overhead_bytes_per_device": GIB,
        "activation_reserve_bytes_per_device": 2 * GIB,
        "max_num_seqs": 64,
        "block_size_tokens": 16,
        "supported_vendors": ("nvidia", "amd"),
    }
    runtime_values.update(runtime_changes)
    values: dict[str, object] = {
        "runtime_id": "runtime-tp1",
        "runtime": RuntimeVariant(**runtime_values),  # type: ignore[arg-type]
        "source": "runtime://pinned",
        "supported_accelerator_ids": ("accelerator-80gb",),
    }
    values.update(changes)
    return RuntimeOption(**values)  # type: ignore[arg-type]


def _providers(*items: ProviderInstanceSpec) -> ProviderInventory:
    return ProviderInventory("provider-inventory-1.0", items)


def _runtimes(*items: RuntimeOption) -> RuntimeInventory:
    return RuntimeInventory("runtime-inventory-1.0", items)


def _load(**changes: object) -> LoadRequirement:
    values: dict[str, object] = {
        "schema_version": "load-requirement-1.0",
        "context_tokens": 8192,
        "peak_concurrent_sequences": 200,
        "target_utilization": 0.8,
    }
    values.update(changes)
    return LoadRequirement(**values)  # type: ignore[arg-type]


def _profile(fingerprint: str) -> MeasuredGroupProfile:
    metrics: dict[str, int | float | str] = {
        "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
        "recipe_variant_fingerprint": fingerprint,
        "context_tokens": 8192,
        "input_tokens_per_request": 100,
        "output_tokens_per_request": 20,
        "requests_per_second": 10,
        "prefill_tokens_per_second": 1000,
        "decode_tokens_per_second": 200,
        "concurrent_sequences": 40,
        "ttft_ms": 100,
        "latency_percentile": 99,
    }
    return MeasuredGroupProfile(
        profile_id="measured-candidate",
        recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
        recipe_variant_fingerprint=fingerprint,
        context_tokens=8192,
        input_tokens_per_request=100,
        output_tokens_per_request=20,
        requests_per_second=10,
        prefill_tokens_per_second=1000,
        decode_tokens_per_second=200,
        concurrent_sequences=40,
        ttft_ms=100,
        latency_percentile=99,
        evidence=(
            EvidenceRecord(
                evidence_id="benchmark",
                kind=EvidenceKind.MEASURED,
                source="benchmark://candidate",
                scope="exact candidate and operating point",
                metrics=metrics,
            ),
        ),
    )


class InventoryTests(unittest.TestCase):
    def test_provider_boundary_rejects_coercion_and_unscoped_price(self) -> None:
        document = _provider().to_dict()
        document["memory_utilization_limit"] = "0.9"
        with self.assertRaisesRegex(ContractError, "memory_utilization_limit"):
            ProviderInstanceSpec.from_dict(document)

        document = _provider().to_dict()
        document["collected_at"] = "2026-08-12"
        with self.assertRaisesRegex(ContractError, "RFC 3339"):
            ProviderInstanceSpec.from_dict(document)

        with self.assertRaisesRegex(ContractError, "price_currency"):
            _provider(price_currency=None)
        with self.assertRaisesRegex(ContractError, "price_currency"):
            _provider(price_currency=123)

    def test_inventory_rejects_duplicate_entries_and_runtime_ids(self) -> None:
        provider = _provider()
        with self.assertRaisesRegex(ContractError, "duplicate provider"):
            _providers(provider, provider)
        runtime = _runtime()
        with self.assertRaisesRegex(ContractError, "duplicate runtime_id"):
            _runtimes(runtime, runtime)

        profile = _profile("sha256:" + "a" * 64)
        with self.assertRaisesRegex(ContractError, "duplicate profile_id"):
            MeasurementInventory("measurement-inventory-1.0", (profile, profile))

    def test_runtime_options_are_engine_agnostic_and_ep_aware(self) -> None:
        option = _runtime(
            runtime={
                "engine": "sglang",
                "tensor_parallel_size": 8,
                "kv_heads_per_device": 1,
                "weight_bytes_per_device_override": 10 * GIB,
            },
            expert_parallel_size=8,
        )
        self.assertEqual(option.runtime.engine, "sglang")
        self.assertEqual(option.expert_parallel_size, 8)

        with self.assertRaisesRegex(ContractError, "divide tensor_parallel_size"):
            replace(option, expert_parallel_size=3)


class PlannerTests(unittest.TestCase):
    def test_planner_accepts_a_replayable_model_manifest(self) -> None:
        manifest = import_huggingface_manifest(
            "example/7b",
            "a" * 40,
            {
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "num_attention_heads": 8,
                "hidden_size": 1024,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16",
                "num_parameters": 7_000_000_000,
            },
            {"metadata": {"total_size": 14_000_000_000}},
        )

        result = explore(
            manifest,
            _providers(_provider()),
            _runtimes(_runtime()),
            context_tokens=8192,
        )

        self.assertEqual(result.model, manifest.model)
        self.assertEqual(result.model_manifest, manifest)
        self.assertEqual(result.candidates[0].status, CandidateStatus.FITS)
        self.assertIn(manifest.evidence[0], result.candidates[0].single_group_audit.recipe.evidence)  # type: ignore[union-attr]

        planned = plan(manifest, _providers(_provider()), _runtimes(_runtime()), _load())
        self.assertEqual(planned.model_manifest, manifest)
        with self.assertRaisesRegex(ContractError, "model_manifest does not match model"):
            replace(result, model=_model(revision="sha256:other"))
        with self.assertRaisesRegex(ContractError, "model_manifest does not match model"):
            replace(planned, model=_model(revision="sha256:other"))

    def test_explore_returns_fits_and_explicit_rejections(self) -> None:
        compatible = _provider()
        unsupported = _provider(
            provider_id="provider-b",
            instance_type="accelerator-other-x1",
            accelerator_id="accelerator-other",
        )

        result = explore(_model(), _providers(compatible, unsupported), _runtimes(_runtime()), context_tokens=8192)

        self.assertEqual(result.candidates[0].status, CandidateStatus.FITS)
        self.assertIsNotNone(result.candidates[0].recipe_variant_fingerprint)
        self.assertEqual(result.candidates[1].status, CandidateStatus.REJECTED)
        self.assertIn("does not support", " ".join(result.candidates[1].rejection_reasons))

    def test_plan_accounts_for_instance_packing_cost_and_idle_devices(self) -> None:
        single = _provider()
        packed = _provider(
            provider_id="provider-b",
            instance_type="accelerator-80gb-x8",
            devices_per_instance=8,
            price_per_instance_hour=10.0,
            max_instances=2,
        )

        result = plan(_model(), _providers(single, packed), _runtimes(_runtime()), _load())

        best = result.candidates[0]
        self.assertEqual(best.provider.provider_id, "provider-b")
        self.assertEqual(best.status, CandidateStatus.FEASIBLE)
        self.assertEqual(best.resource_claim.required_instances, 1)  # type: ignore[union-attr]
        self.assertEqual(best.resource_claim.serving_replicas, 5)  # type: ignore[union-attr]
        self.assertEqual(best.resource_claim.allocated_devices, 8)  # type: ignore[union-attr]
        self.assertEqual(best.resource_claim.idle_devices, 3)  # type: ignore[union-attr]
        self.assertEqual(best.hourly_cost, 10.0)

        fewest = plan(
            _model(),
            _providers(single, packed),
            _runtimes(_runtime()),
            _load(),
            objective=PlanningObjective.FEWEST_ALLOCATED_DEVICES,
        )
        self.assertEqual(fewest.candidates[0].provider.provider_id, "provider-a")

    def test_plan_marks_unknown_performance_and_limited_inventory(self) -> None:
        traffic = _load(
            peak_concurrent_sequences=0,
            request_rate_per_second=8,
            input_tokens_per_request=100,
            output_tokens_per_request=20,
        )
        incomplete = plan(_model(), _providers(_provider()), _runtimes(_runtime()), traffic)
        self.assertEqual(incomplete.candidates[0].status, CandidateStatus.INCOMPLETE)
        self.assertIsNone(incomplete.candidates[0].resource_claim)

        limited = plan(
            _model(),
            _providers(_provider(max_instances=2)),
            _runtimes(_runtime()),
            _load(),
        )
        self.assertEqual(limited.candidates[0].status, CandidateStatus.INSUFFICIENT_INVENTORY)
        self.assertFalse(limited.candidates[0].resource_claim.fits_available_inventory)  # type: ignore[union-attr]

    def test_measurement_inventory_enables_traffic_and_slo_planning(self) -> None:
        provider = _provider()
        runtime = _runtime()
        explored = explore(_model(), _providers(provider), _runtimes(runtime), context_tokens=8192)
        fingerprint = explored.candidates[0].recipe_variant_fingerprint
        assert fingerprint is not None
        measurements = MeasurementInventory("measurement-inventory-1.0", (_profile(fingerprint),))
        load = _load(
            peak_concurrent_sequences=0,
            request_rate_per_second=8,
            input_tokens_per_request=100,
            output_tokens_per_request=20,
            target_ttft_ms=150,
            target_latency_percentile=99,
        )

        result = plan(
            _model(),
            _providers(provider),
            _runtimes(runtime),
            load,
            measurements=measurements,
        )

        candidate = result.candidates[0]
        self.assertEqual(candidate.status, CandidateStatus.FEASIBLE)
        self.assertEqual(candidate.confidence, "analytical-memory+measured-performance")
        self.assertEqual(candidate.resource_claim.serving_replicas, 1)  # type: ignore[union-attr]

    def test_quantization_changes_fit_without_changing_model_identity_rules(self) -> None:
        provider = _provider(
            instance_type="accelerator-16gb-x1",
            accelerator_id="accelerator-16gb",
            memory_bytes_per_device=16 * GIB,
            price_per_instance_hour=None,
            price_currency=None,
            max_instances=None,
            collected_at=None,
        )
        runtime = _runtime(supported_accelerator_ids=("accelerator-16gb",))
        bf16 = explore(_model(max_model_len=1024), _providers(provider), _runtimes(runtime), context_tokens=1024)
        int4 = explore(
            _model(weight_dtype="int4", max_model_len=1024),
            _providers(provider),
            _runtimes(runtime),
            context_tokens=1024,
        )
        self.assertEqual(bf16.candidates[0].status, CandidateStatus.REJECTED)
        self.assertEqual(int4.candidates[0].status, CandidateStatus.FITS)

    def test_quantized_manifest_requires_measured_runtime_resident_weight_bytes(self) -> None:
        manifest = import_huggingface_manifest(
            "example/quantized",
            "a" * 40,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 1024,
                "quantization_config": {"bits": 4},
            },
            {"metadata": {"total_size": 4 * GIB}},
            parameter_count_override=7_000_000_000,
        )
        provider = _provider(
            instance_type="accelerator-16gb-x1",
            accelerator_id="accelerator-16gb",
            memory_bytes_per_device=16 * GIB,
            price_per_instance_hour=None,
            price_currency=None,
            max_instances=None,
            collected_at=None,
        )
        runtime = _runtime(supported_accelerator_ids=("accelerator-16gb",))

        unresolved = explore(manifest, _providers(provider), _runtimes(runtime), context_tokens=1024)
        measured_total_manifest = replace(
            manifest,
            model=replace(manifest.model, explicit_weight_bytes=4 * GIB),
        )
        measured_total = explore(
            measured_total_manifest,
            _providers(provider),
            _runtimes(runtime),
            context_tokens=1024,
        )
        mixed_manifest = replace(
            measured_total_manifest,
            model=replace(measured_total_manifest.model, weight_dtype="quantized"),
        )
        multi_device = explore(
            mixed_manifest,
            _providers(replace(provider, devices_per_instance=8)),
            _runtimes(
                replace(
                    runtime,
                    runtime=replace(
                        runtime.runtime,
                        tensor_parallel_size=8,
                        kv_heads_per_device=1,
                    ),
                )
            ),
            context_tokens=1024,
        )
        measured = explore(
            manifest,
            _providers(provider),
            _runtimes(
                replace(
                    runtime,
                    runtime=replace(runtime.runtime, weight_bytes_per_device_override=4 * GIB),
                )
            ),
            context_tokens=1024,
        )

        self.assertEqual(unresolved.candidates[0].status, CandidateStatus.REJECTED)
        self.assertIn("measured per-device resident weight bytes", unresolved.candidates[0].rejection_reasons[0])
        self.assertEqual(measured_total.candidates[0].status, CandidateStatus.FITS)
        self.assertEqual(multi_device.candidates[0].status, CandidateStatus.REJECTED)
        self.assertEqual(measured.candidates[0].status, CandidateStatus.FITS)

    def test_flow_control_is_used_by_provider_plans(self) -> None:
        provider = _provider(
            instance_type="accelerator-288gb-x8",
            devices_per_instance=8,
            memory_bytes_per_device=288 * GIB,
            price_per_instance_hour=32.0,
            max_instances=20,
        )
        model = ModelSpec(
            model_id="example/long-context-moe",
            revision="sha256:fixture",
            parameter_count=240_000_000_000,
            num_layers=64,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype="bf16",
            max_model_len=1_048_576,
            attention_type="mla",
            kv_bytes_per_token_per_device_override=4096,
        )
        runtime = _runtime(
            runtime_id="runtime-tp8-ep8",
            runtime={
                "tensor_parallel_size": 8,
                "kv_cache_dtype": "fp8",
                "weight_bytes_per_device_override": 60 * GIB,
                "max_num_seqs": 24,
                "block_size_tokens": 1536,
            },
            expert_parallel_size=8,
            routing=LLMDRoutingSpec(
                flow_control_token_limit=2_104_030,
                kv_block_size_tokens=1536,
                precise_prefix_routing=True,
            ),
        )

        result = plan(
            model,
            _providers(provider),
            _runtimes(runtime),
            _load(context_tokens=1_048_576, peak_concurrent_sequences=30),
        )

        candidate = result.candidates[0]
        self.assertEqual(candidate.status, CandidateStatus.FEASIBLE)
        self.assertEqual(candidate.single_group_audit.status, AuditStatus.INSUFFICIENT)  # type: ignore[union-attr]
        self.assertEqual(candidate.sequences_per_replica, 2)
        self.assertEqual(candidate.resource_claim.serving_replicas, 19)  # type: ignore[union-attr]
        self.assertEqual(candidate.resource_claim.allocated_devices, 152)  # type: ignore[union-attr]
        self.assertTrue(any("not divisible" in warning for warning in candidate.warnings))
        self.assertEqual(sum("not divisible" in warning for warning in candidate.warnings), 1)

    def test_lowest_cost_rejects_mixed_currencies(self) -> None:
        euro = _provider(provider_id="provider-eu", price_currency="EUR")
        with self.assertRaisesRegex(ContractError, "different currencies"):
            plan(_model(), _providers(_provider(), euro), _runtimes(_runtime()), _load())
        result = plan(
            _model(),
            _providers(_provider(), euro),
            _runtimes(_runtime()),
            _load(),
            objective=PlanningObjective.FEWEST_ALLOCATED_DEVICES,
        )
        self.assertEqual(len(result.candidates), 2)

    def test_more_memory_never_reduces_explored_sequence_capacity(self) -> None:
        capacities: list[int] = []
        for memory_gib in range(20, 121, 5):
            provider = _provider(
                instance_type=f"accelerator-{memory_gib}gb-x1",
                memory_bytes_per_device=memory_gib * GIB,
                price_per_instance_hour=None,
                price_currency=None,
                max_instances=None,
                collected_at=None,
            )
            candidate = explore(
                _model(),
                _providers(provider),
                _runtimes(_runtime()),
                context_tokens=8192,
            ).candidates[0]
            capacities.append(candidate.sequences_per_replica)
        self.assertEqual(capacities, sorted(capacities))

    def test_required_replicas_are_monotone_in_concurrency_demand(self) -> None:
        replicas: list[int] = []
        for demand in range(1, 501, 13):
            candidate = plan(
                _model(),
                _providers(_provider(max_instances=None)),
                _runtimes(_runtime()),
                _load(peak_concurrent_sequences=demand),
                objective=PlanningObjective.FEWEST_ALLOCATED_DEVICES,
            ).candidates[0]
            assert candidate.resource_claim is not None
            replicas.append(candidate.resource_claim.serving_replicas)
        self.assertEqual(replicas, sorted(replicas))


if __name__ == "__main__":
    unittest.main()
