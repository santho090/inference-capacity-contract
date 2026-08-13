from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from inference_capacity_contract import (
    RECIPE_VARIANT_FINGERPRINT_VERSION,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    LoadRequirement,
    MeasuredGroupProfile,
    MeasurementInventory,
    ModelManifest,
    ModelPlanningResult,
    ModelPlanningStatus,
    ModelSpec,
    PlanningObjective,
    ProviderInventory,
    RuntimeInventory,
    RuntimeVariant,
    ServingRecipe,
    WorkloadProfile,
    audit_recipe,
    audit_recipe_draft,
    capacity_for,
    explore,
    import_huggingface_manifest,
    import_llmd_values,
    import_vllm_initialization,
    inspect_huggingface_config,
    plan,
    plan_huggingface_model,
    recommend_scale,
)

ROOT = Path(__file__).parents[1]
GIB = 1024**3


def _read_schema(name: str) -> dict[str, object]:
    value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def _read_fixture(name: str) -> dict[str, object]:
    value = json.loads((ROOT / "docs" / "fixtures" / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity_schema = _read_schema("capacity-contract-2.0.schema.json")
        self.scaling_schema = _read_schema("scaling-recommendation-2.0.schema.json")
        self.recipe_schema = _read_schema("serving-recipe-1.0.schema.json")
        self.load_schema = _read_schema("load-requirement-1.0.schema.json")
        self.audit_schema = _read_schema("recipe-audit-2.0.schema.json")
        self.draft_schema = _read_schema("recipe-draft-1.0.schema.json")
        self.structural_audit_schema = _read_schema("structural-recipe-audit-1.0.schema.json")
        self.manifest_schema = _read_schema("model-manifest-1.0.schema.json")
        self.model_resolution_schema = _read_schema("model-resolution-draft-1.0.schema.json")
        self.initialization_schema = _read_schema("vllm-initialization-profile-1.0.schema.json")
        self.provider_inventory_schema = _read_schema("provider-inventory-1.0.schema.json")
        self.runtime_inventory_schema = _read_schema("runtime-inventory-1.0.schema.json")
        self.measurement_inventory_schema = _read_schema("measurement-inventory-1.0.schema.json")
        self.exploration_schema = _read_schema("capacity-exploration-1.0.schema.json")
        self.plan_schema = _read_schema("capacity-plan-1.0.schema.json")
        self.model_planning_schema = _read_schema("model-planning-result-1.0.schema.json")
        Draft202012Validator.check_schema(self.capacity_schema)
        Draft202012Validator.check_schema(self.scaling_schema)
        Draft202012Validator.check_schema(self.recipe_schema)
        Draft202012Validator.check_schema(self.load_schema)
        Draft202012Validator.check_schema(self.audit_schema)
        Draft202012Validator.check_schema(self.draft_schema)
        Draft202012Validator.check_schema(self.structural_audit_schema)
        Draft202012Validator.check_schema(self.manifest_schema)
        Draft202012Validator.check_schema(self.model_resolution_schema)
        Draft202012Validator.check_schema(self.initialization_schema)
        Draft202012Validator.check_schema(self.provider_inventory_schema)
        Draft202012Validator.check_schema(self.runtime_inventory_schema)
        Draft202012Validator.check_schema(self.measurement_inventory_schema)
        Draft202012Validator.check_schema(self.exploration_schema)
        Draft202012Validator.check_schema(self.plan_schema)
        Draft202012Validator.check_schema(self.model_planning_schema)

    def test_model_planning_result_document_validates(self) -> None:
        providers = ProviderInventory.from_dict(_read_fixture("provider-inventory.json"))
        runtimes = RuntimeInventory.from_dict(_read_fixture("runtime-inventory.json"))
        load = LoadRequirement.from_dict(_read_fixture("load-provider-plan.json"))

        def fetch(url: str) -> dict[str, object]:
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 8192,
                    "torch_dtype": "float16",
                    "num_parameters": 1024,
                }
            return {"metadata": {"total_size": 2048}}

        result = plan_huggingface_model(
            "example/model",
            providers,
            runtimes,
            load,
            revision="a" * 40,
            fetch_json=fetch,
            inspect_safetensors_headers=False,
        )
        self.assertEqual(result.status, ModelPlanningStatus.PLANNED)

        registry = Registry()
        for schema in (
            self.capacity_schema,
            self.recipe_schema,
            self.load_schema,
            self.audit_schema,
            self.provider_inventory_schema,
            self.runtime_inventory_schema,
            self.manifest_schema,
            self.exploration_schema,
            self.plan_schema,
            self.model_resolution_schema,
            self.model_planning_schema,
        ):
            registry = registry.with_resource(str(schema["$id"]), Resource.from_contents(schema))
        Draft202012Validator(self.model_planning_schema, registry=registry).validate(result.to_dict())
        unresolved = ModelPlanningResult(
            "model-planning-result-1.0",
            ModelPlanningStatus.NEEDS_MODEL_INPUTS,
            inspect_huggingface_config(
                "example/model",
                "a" * 40,
                {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 8192,
                    "torch_dtype": "float16",
                },
            ),
            None,
        )
        Draft202012Validator(self.model_planning_schema, registry=registry).validate(unresolved.to_dict())

    def test_provider_inventory_exploration_and_plan_documents_validate(self) -> None:
        model = ModelSpec.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "model-7b.json").read_text(encoding="utf-8"))
        )
        providers = ProviderInventory.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "provider-inventory.json").read_text(encoding="utf-8"))
        )
        runtimes = RuntimeInventory.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "runtime-inventory.json").read_text(encoding="utf-8"))
        )
        load = LoadRequirement.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "load-provider-plan.json").read_text(encoding="utf-8"))
        )
        exploration = explore(model, providers, runtimes, context_tokens=load.context_tokens).to_dict()
        capacity_plan = plan(
            model,
            providers,
            runtimes,
            load,
            objective=PlanningObjective.LOWEST_COST,
        ).to_dict()

        registry = Registry()
        for schema in (
            self.capacity_schema,
            self.recipe_schema,
            self.load_schema,
            self.audit_schema,
            self.provider_inventory_schema,
            self.runtime_inventory_schema,
            self.measurement_inventory_schema,
            self.manifest_schema,
            self.exploration_schema,
            self.plan_schema,
        ):
            registry = registry.with_resource(str(schema["$id"]), Resource.from_contents(schema))

        Draft202012Validator(self.provider_inventory_schema, registry=registry).validate(providers.to_dict())
        Draft202012Validator(self.runtime_inventory_schema, registry=registry).validate(runtimes.to_dict())
        Draft202012Validator(self.measurement_inventory_schema, registry=registry).validate(
            MeasurementInventory("measurement-inventory-1.0", ()).to_dict()
        )
        Draft202012Validator(self.exploration_schema, registry=registry).validate(exploration)
        Draft202012Validator(self.plan_schema, registry=registry).validate(capacity_plan)

        manifest = ModelManifest.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "model-manifest-long-context.json").read_text(encoding="utf-8"))
        )
        Draft202012Validator(self.manifest_schema, registry=registry).validate(manifest.to_dict())
        manifest_exploration = explore(manifest, providers, runtimes, context_tokens=load.context_tokens).to_dict()
        Draft202012Validator(self.exploration_schema, registry=registry).validate(manifest_exploration)
        self.assertEqual(manifest_exploration["model_manifest"], manifest.to_dict())

    def test_model_resolution_draft_document_validates(self) -> None:
        draft = inspect_huggingface_config(
            "example/model",
            "a" * 40,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 1024,
                "torch_dtype": "float16",
            },
        )

        Draft202012Validator(self.model_resolution_schema).validate(draft.to_dict())

    def test_draft_manifest_and_initialization_documents_validate(self) -> None:
        values = json.loads(
            (ROOT / "docs" / "fixtures" / "llmd-values-long-context-tp8.json").read_text(encoding="utf-8")
        )
        hardware = HardwareSpec.from_dict(
            json.loads((ROOT / "docs" / "fixtures" / "hardware-accelerator-x8-288gb.json").read_text(encoding="utf-8"))
        )
        draft = import_llmd_values(values, host=hardware)
        structural = audit_recipe_draft(draft)
        manifest = import_huggingface_manifest(
            "example/model",
            "a" * 40,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 1024,
                "torch_dtype": "float16",
                "num_parameters": 1024,
            },
            {"metadata": {"total_size": 2048}},
        )
        initialization = import_vllm_initialization(
            {
                "model_revision": "a" * 40,
                "hardware_id": "h100",
                "runtime_engine": "vllm",
                "runtime_version": "0.8.5",
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
                "physical_device_count": 1,
                "memory_utilization_limit": 0.9,
                "kv_cache_dtype": "bf16",
                "block_size_tokens": 16,
                "weight_bytes_per_device": 2048,
                "runtime_overhead_bytes_per_device": 0,
                "activation_reserve_bytes_per_device": 0,
                "kv_bytes_per_token_per_device": 512,
                "kv_capacity_tokens_per_device": 1000,
            },
            source="benchmark://schema-init",
        )
        registry = Registry().with_resource(
            str(self.capacity_schema["$id"]), Resource.from_contents(self.capacity_schema)
        )
        Draft202012Validator(self.draft_schema).validate(draft.to_dict())
        Draft202012Validator(self.structural_audit_schema).validate(structural.to_dict())
        Draft202012Validator(self.manifest_schema, registry=registry).validate(manifest.to_dict())
        Draft202012Validator(self.initialization_schema, registry=registry).validate(initialization.to_dict())

    def test_recipe_load_and_audit_documents_validate(self) -> None:
        recipe_document = json.loads(
            (ROOT / "docs" / "fixtures" / "serving-recipe-hybrid-tp8.json").read_text(encoding="utf-8")
        )
        load_document = json.loads(
            (ROOT / "docs" / "fixtures" / "load-context-concurrency.json").read_text(encoding="utf-8")
        )
        recipe = ServingRecipe.from_dict(recipe_document)
        load = LoadRequirement.from_dict(load_document)
        audit = audit_recipe(recipe, load).to_dict()

        registry = Registry()
        for schema in (self.capacity_schema, self.recipe_schema, self.load_schema):
            registry = registry.with_resource(str(schema["$id"]), Resource.from_contents(schema))
        Draft202012Validator(self.recipe_schema, registry=registry).validate(recipe.to_dict())
        Draft202012Validator(self.load_schema, registry=registry).validate(load.to_dict())
        Draft202012Validator(self.audit_schema, registry=registry).validate(audit)
        self.assertEqual(audit["schema_version"], "recipe-audit-2.0")
        self.assertEqual(audit["formula_version"], "recipe-audit-formula-2.0")
        self.assertEqual(audit["effective_sequences_per_group"], 24)
        self.assertEqual(audit["status"], "sufficient")

        measured = MeasuredGroupProfile(
            profile_id="schema-profile",
            recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
            recipe_variant_fingerprint=recipe.variant_fingerprint,
            context_tokens=load.context_tokens,
            requests_per_second=5,
            ttft_ms=100,
            latency_percentile=99,
            evidence=(
                EvidenceRecord(
                    evidence_id="schema-measurement",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://schema-measurement",
                    scope="exact recipe fingerprint at the recorded operating point",
                    metrics={
                        "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
                        "recipe_variant_fingerprint": recipe.variant_fingerprint,
                        "context_tokens": load.context_tokens,
                        "input_tokens_per_request": 0,
                        "output_tokens_per_request": 0,
                        "requests_per_second": 5,
                        "ttft_ms": 100,
                        "latency_percentile": 99,
                    },
                ),
            ),
        )
        measured_load = replace(
            load,
            request_rate_per_second=4,
            target_ttft_ms=150,
            target_latency_percentile=99,
            measured_profile=measured,
        )
        measured_audit = audit_recipe(recipe, measured_load).to_dict()
        Draft202012Validator(self.load_schema, registry=registry).validate(measured_load.to_dict())
        Draft202012Validator(self.audit_schema, registry=registry).validate(measured_audit)

    def test_generated_contract_and_recommendation_validate(self) -> None:
        contract = capacity_for(
            ModelSpec(
                model_id="example/7b",
                revision="sha256:example",
                parameter_count=7_000_000_000,
                num_layers=32,
                num_kv_heads=8,
                head_dim=128,
                max_model_len=8192,
            ),
            HardwareSpec(
                "h100",
                "nvidia",
                1,
                80 * GIB,
                price_per_device_hour=2.0,
            ),
            RuntimeVariant(
                "vllm",
                "0.8.5",
                runtime_overhead_bytes_per_device=GIB,
                activation_reserve_bytes_per_device=2 * GIB,
            ),
        )
        profile = WorkloadProfile(
            profile_id="profile",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=5,
            sustainable_requests_per_replica_per_second=5,
            evidence=(
                EvidenceRecord(
                    evidence_id="run",
                    kind=EvidenceKind.MEASURED,
                    source="benchmark://run",
                    scope="exact variant",
                ),
            ),
        )
        recommendation = recommend_scale(contract, profile)
        capacity_validator = Draft202012Validator(self.capacity_schema)
        capacity_validator.validate(contract.to_dict())

        capacity_resource = Resource.from_contents(self.capacity_schema)
        capacity_schema_id = str(self.capacity_schema["$id"])
        registry = Registry().with_resource(capacity_schema_id, capacity_resource)
        scaling_validator = Draft202012Validator(
            self.scaling_schema,
            registry=registry,
        )
        scaling_validator.validate(recommendation.to_dict())

    def test_schema_rejects_empty_nested_contract_objects(self) -> None:
        invalid = {
            "schema_version": "capacity-contract-2.0",
            "model": {},
            "hardware": {},
            "runtime": {},
            "fits": True,
            "validation_level": "analytically-feasible",
            "memory_bytes_per_device": {},
            "kv_bytes_per_token_per_device": 1,
            "kv_block_size_tokens": 16,
            "kv_capacity_blocks_per_device": 0,
            "kv_capacity_tokens_per_device": 0,
            "max_context_tokens": 0,
            "concurrency_envelope": [],
            "assumptions": [],
            "warnings": [],
            "evidence": [],
        }
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.capacity_schema).validate(invalid)

    def test_schema_requires_tp_kv_layout_and_complete_concurrency_measurement(self) -> None:
        contract = capacity_for(
            ModelSpec(
                model_id="example/7b",
                revision="sha256:example",
                parameter_count=7_000_000_000,
                num_layers=32,
                num_kv_heads=8,
                head_dim=128,
                max_model_len=8192,
            ),
            HardwareSpec("h100", "nvidia", 1, 80 * GIB),
            RuntimeVariant("vllm", "0.8.5"),
        ).to_dict()
        contract["runtime"]["tensor_parallel_size"] = 2
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.capacity_schema).validate(contract)

        profile = WorkloadProfile(
            profile_id="profile",
            model_id="example/7b",
            model_revision="sha256:example",
            hardware_id="h100",
            runtime_engine="vllm",
            runtime_version="0.8.5",
            request_rate_per_second=1,
            sustainable_requests_per_replica_per_second=1,
        ).to_dict()
        profile["peak_concurrent_sequences"] = 2
        recommendation = {
            "schema_version": "scaling-recommendation-2.0",
            "contract": capacity_for(
                ModelSpec(
                    model_id="example/7b",
                    revision="sha256:example",
                    parameter_count=7_000_000_000,
                    num_layers=32,
                    num_kv_heads=8,
                    head_dim=128,
                    max_model_len=8192,
                ),
                HardwareSpec("h100", "nvidia", 1, 80 * GIB),
                RuntimeVariant("vllm", "0.8.5"),
            ).to_dict(),
            "profile": profile,
            "required_replicas": 1,
            "recommended_replicas": 1,
            "drivers": {"requests_per_second": 1},
            "constrained_by": "requests_per_second",
            "within_bounds": True,
            "estimated_gpu_hours_per_hour": 1,
            "estimated_gpu_hour_savings_vs_baseline": None,
            "estimated_hourly_cost": None,
            "estimated_hourly_savings_vs_baseline": None,
            "suggestions": [],
            "warnings": [],
        }
        capacity_resource = Resource.from_contents(self.capacity_schema)
        registry = Registry().with_resource(str(self.capacity_schema["$id"]), capacity_resource)
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.scaling_schema, registry=registry).validate(recommendation)


if __name__ == "__main__":
    unittest.main()
