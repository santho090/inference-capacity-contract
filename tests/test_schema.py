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
    ModelSpec,
    RuntimeVariant,
    ServingRecipe,
    WorkloadProfile,
    audit_recipe,
    capacity_for,
    recommend_scale,
)

ROOT = Path(__file__).parents[1]
GIB = 1024**3


def _read_schema(name: str) -> dict[str, object]:
    value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


class SchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.capacity_schema = _read_schema("capacity-contract-2.0.schema.json")
        self.scaling_schema = _read_schema("scaling-recommendation-2.0.schema.json")
        self.recipe_schema = _read_schema("serving-recipe-1.0.schema.json")
        self.load_schema = _read_schema("load-requirement-1.0.schema.json")
        self.audit_schema = _read_schema("recipe-audit-1.0.schema.json")
        Draft202012Validator.check_schema(self.capacity_schema)
        Draft202012Validator.check_schema(self.scaling_schema)
        Draft202012Validator.check_schema(self.recipe_schema)
        Draft202012Validator.check_schema(self.load_schema)
        Draft202012Validator.check_schema(self.audit_schema)

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
