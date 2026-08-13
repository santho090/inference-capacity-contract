from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from inference_capacity_contract import (
    ContractError,
    LoadRequirement,
    ModelPlanningResult,
    ModelPlanningStatus,
    ProviderInventory,
    RuntimeInventory,
    plan_huggingface_model,
)

ROOT = Path(__file__).parents[1]
PINNED_REVISION = "a" * 40


def _fixture(name: str) -> dict[str, object]:
    value = json.loads((ROOT / "docs" / "fixtures" / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain an object")
    return value


class ModelPlanningWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = ProviderInventory.from_dict(_fixture("provider-inventory.json"))
        self.runtimes = RuntimeInventory.from_dict(_fixture("runtime-inventory.json"))
        self.load = LoadRequirement.from_dict(_fixture("load-provider-plan.json"))

    def test_returns_actionable_model_inputs_before_provider_planning(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if "/api/models/" in url:
                return {"id": "example/hybrid", "sha": PINNED_REVISION}
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 8192,
                    "linear_attn_config": {"head_dim": 64},
                    "quantization_config": {"bits": 4, "ignore": ["lm_head"]},
                }
            self.fail(f"incomplete model fetched weight metadata: {url}")

        result = plan_huggingface_model(
            "example/hybrid",
            self.providers,
            self.runtimes,
            self.load,
            fetch_json=fetch,
        )

        self.assertEqual(result.status, ModelPlanningStatus.NEEDS_MODEL_INPUTS)
        self.assertIsNone(result.capacity_plan)
        assert result.model_resolution is not None
        self.assertEqual(result.model_resolution.revision, PINNED_REVISION)
        self.assertEqual(
            {item.path for item in result.model_resolution.unresolved},
            {
                "parameter_count_override",
                "resident_weight_bytes_override",
            },
        )
        self.assertEqual(len(calls), 2)

    def test_resolves_complete_model_and_returns_ranked_plan(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
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
            if url.endswith("model.safetensors.index.json"):
                return {"metadata": {"total_size": 2048}}
            self.fail(f"unexpected model metadata request: {url}")

        result = plan_huggingface_model(
            "example/model",
            self.providers,
            self.runtimes,
            self.load,
            revision=PINNED_REVISION,
            fetch_json=fetch,
            inspect_safetensors_headers=False,
        )

        self.assertEqual(result.status, ModelPlanningStatus.PLANNED)
        self.assertIsNone(result.model_resolution)
        assert result.capacity_plan is not None
        self.assertEqual(result.capacity_plan.model.revision, PINNED_REVISION)
        self.assertEqual(len(result.capacity_plan.candidates), 3)
        self.assertEqual(result.capacity_plan.candidates[0].rank, 1)
        self.assertEqual(len(calls), 2)

        with self.assertRaisesRegex(ContractError, "replayable model manifest"):
            ModelPlanningResult(
                "model-planning-result-1.0",
                ModelPlanningStatus.PLANNED,
                None,
                replace(result.capacity_plan, model_manifest=None),
            )

    def test_complete_hybrid_model_still_requires_runtime_capacity(self) -> None:
        def fetch(url: str) -> dict[str, object]:
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 2,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 8192,
                    "linear_attn_config": {"head_dim": 64},
                    "quantization_config": {"bits": 4, "ignore": ["lm_head"]},
                }
            if url.endswith("model.safetensors.index.json"):
                return {"metadata": {"total_size": 2048}}
            self.fail(f"unexpected model metadata request: {url}")

        result = plan_huggingface_model(
            "example/hybrid",
            self.providers,
            self.runtimes,
            self.load,
            revision=PINNED_REVISION,
            parameter_count_override=1000,
            resident_weight_bytes_override=2048,
            fetch_json=fetch,
            inspect_safetensors_headers=False,
        )

        self.assertEqual(result.status, ModelPlanningStatus.PLANNED)
        assert result.capacity_plan is not None
        self.assertTrue(result.capacity_plan.candidates)
        for candidate in result.capacity_plan.candidates:
            self.assertIn("runtime KV capacity envelope", " ".join(candidate.rejection_reasons))

    def test_result_rejects_mixed_or_empty_states(self) -> None:
        with self.assertRaisesRegex(ContractError, "requires one unresolved"):
            ModelPlanningResult(
                "model-planning-result-1.0",
                ModelPlanningStatus.NEEDS_MODEL_INPUTS,
                None,
                None,
            )
        with self.assertRaisesRegex(ContractError, "ModelResolutionDraft"):
            ModelPlanningResult(
                "model-planning-result-1.0",
                ModelPlanningStatus.NEEDS_MODEL_INPUTS,
                "draft",  # type: ignore[arg-type]
                None,
            )
