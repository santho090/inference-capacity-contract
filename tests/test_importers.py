from __future__ import annotations

import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from inference_capacity_contract import (
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    LLMDRoutingSpec,
    ModelManifest,
    ModelResolutionDraft,
    ModelSpec,
    ParallelTopology,
    RuntimeVariant,
    ServingRecipe,
    import_huggingface_manifest,
    import_llmd_values,
    import_vllm_benchmark,
    import_vllm_initialization,
    inspect_huggingface_config,
    materialize_recipe_draft,
    parse_safetensors_header,
    resolve_huggingface_manifest,
    resolve_huggingface_model_draft,
    resolve_huggingface_revision,
    tensor_element_count_from_safetensors_headers,
)

GIB = 1024**3
PINNED_REVISION = "a" * 40


def _init_identity(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
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
    }
    values.update(changes)
    return values


def _recipe() -> ServingRecipe:
    return ServingRecipe(
        schema_version="serving-recipe-1.0",
        recipe_id="benchmark-fixture",
        model=ModelSpec("example/7b", PINNED_REVISION, 7_000_000_000, 32, 8, 128, max_model_len=8192),
        host=HardwareSpec("h100", "nvidia", 1, 80 * GIB),
        runtime=RuntimeVariant(
            "vllm",
            "0.8.5",
            runtime_overhead_bytes_per_device=GIB,
            activation_reserve_bytes_per_device=GIB,
            max_num_seqs=128,
        ),
        topology=ParallelTopology(),
        llmd=LLMDRoutingSpec(),
        evidence=(EvidenceRecord("recipe", EvidenceKind.REPORTED, "fixture://recipe", "exact fixture"),),
    )


class ModelManifestImporterTests(unittest.TestCase):
    def test_resolves_model_references_to_immutable_revisions(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            return {"id": "example/model", "sha": PINNED_REVISION}

        self.assertEqual(resolve_huggingface_revision("example/model", fetch_json=fetch), PINNED_REVISION)
        self.assertEqual(
            calls,
            ["https://huggingface.co/api/models/example/model/revision/main"],
        )
        self.assertEqual(
            resolve_huggingface_revision(
                "example/model",
                PINNED_REVISION.upper(),
                fetch_json=lambda _: self.fail("an immutable revision reached the network"),
            ),
            PINNED_REVISION,
        )

    def test_rejects_invalid_reference_metadata(self) -> None:
        cases = (
            ({"id": "other/model", "sha": PINNED_REVISION}, "requested repository"),
            ({"id": "example/model", "sha": "not-a-sha"}, "invalid commit SHA"),
        )
        for model_info, message in cases:
            with self.subTest(model_info=model_info):
                with self.assertRaisesRegex(ContractError, message):
                    resolve_huggingface_revision(
                        "example/model",
                        "release/v1",
                        fetch_json=lambda _, response=model_info: response,
                    )

        for reference in (" release", "release/", "release/../main"):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ContractError, "valid branch"):
                    resolve_huggingface_revision(
                        "example/model",
                        reference,
                        fetch_json=lambda _: self.fail("invalid reference reached the network"),
                    )

    def test_model_draft_pins_main_and_reuses_model_metadata(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if "/api/models/" in url:
                return {
                    "id": "example/model",
                    "sha": PINNED_REVISION,
                    "safetensors": {"total": 1024},
                }
            if url.endswith(f"/{PINNED_REVISION}/config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                }
            self.fail(f"unexpected metadata request: {url}")

        draft = resolve_huggingface_model_draft("example/model", fetch_json=fetch)

        self.assertEqual(draft.revision, PINNED_REVISION)
        self.assertEqual(draft.field_sources["revision"], "huggingface-api.sha")
        self.assertEqual(draft.facts["parameter_count"], 1024)
        self.assertEqual(len(calls), 2)

    def test_model_resolution_draft_preserves_discovered_kimi_facts(self) -> None:
        config: dict[str, object] = {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 93,
                "num_key_value_heads": 96,
                "num_attention_heads": 96,
                "hidden_size": 7168,
                "v_head_dim": 128,
                "max_position_embeddings": 1_048_576,
                "kv_lora_rank": 512,
                "linear_attn_config": {"head_dim": 128},
                "quantization_config": {
                    "format": "mxfp4-pack-quantized",
                    "ignore": ["lm_head"],
                    "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
                },
            },
        }

        draft = inspect_huggingface_config("moonshotai/Kimi-K3", PINNED_REVISION, config)
        restored = ModelResolutionDraft.from_dict(draft.to_dict())

        self.assertEqual(restored, draft)
        self.assertFalse(draft.ready)
        self.assertEqual(draft.facts["num_layers"], 93)
        self.assertEqual(draft.facts["num_kv_heads"], 96)
        self.assertEqual(draft.facts["head_dim"], 128)
        self.assertEqual(draft.facts["attention_type"], "hybrid")
        self.assertEqual(draft.facts["weight_dtype"], "quantized")
        self.assertEqual(draft.facts["quantization_format"], "mxfp4-pack-quantized")
        self.assertEqual(
            {item.path for item in draft.unresolved},
            {
                "parameter_count_override",
                "kv_bytes_per_token_per_device_override",
                "resident_weight_bytes_override",
            },
        )

    def test_online_model_draft_uses_api_count_for_unquantized_config(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                }
            return {"sha": PINNED_REVISION, "safetensors": {"total": 1024}}

        draft = resolve_huggingface_model_draft(
            "example/model",
            PINNED_REVISION,
            fetch_json=fetch,
        )

        self.assertTrue(draft.ready)
        self.assertEqual(draft.facts["parameter_count"], 1024)
        self.assertEqual(draft.field_sources["parameter_count"], "huggingface-api.safetensors.total")
        self.assertEqual(len(calls), 2)

    def test_model_draft_reports_missing_architecture_fields_instead_of_crashing(self) -> None:
        draft = inspect_huggingface_config(
            "example/sparse-config",
            PINNED_REVISION,
            {"torch_dtype": "bfloat16", "num_parameters": 1000},
        )

        self.assertFalse(draft.ready)
        self.assertEqual(
            {item.path for item in draft.unresolved},
            {
                "model.num_layers",
                "model.num_kv_heads",
                "model.head_dim",
                "model.max_model_len",
                "model.attention_type",
            },
        )

    def test_model_resolution_parser_rejects_tampered_readiness_and_provenance(self) -> None:
        draft = inspect_huggingface_config(
            "example/model",
            PINNED_REVISION,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 1024,
                "torch_dtype": "float16",
                "num_parameters": 1000,
            },
        )
        readiness = draft.to_dict()
        readiness["ready"] = False
        with self.assertRaisesRegex(ContractError, "ready does not match"):
            ModelResolutionDraft.from_dict(readiness)

        provenance = draft.to_dict()
        del provenance["field_sources"]["head_dim"]
        with self.assertRaisesRegex(ContractError, "missing field provenance"):
            ModelResolutionDraft.from_dict(provenance)

        incomplete = inspect_huggingface_config(
            "example/model",
            PINNED_REVISION,
            {"torch_dtype": "float16"},
        ).to_dict()
        incomplete["unresolved"][0]["required_for"] = "guessing"
        with self.assertRaisesRegex(ContractError, "required_for is unsupported"):
            ModelResolutionDraft.from_dict(incomplete)

    def test_unknown_weight_format_can_be_bound_by_measured_resident_bytes(self) -> None:
        config = {
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_attention_heads": 2,
            "hidden_size": 128,
            "max_position_embeddings": 1024,
            "torch_dtype": "vendor_packed",
            "num_parameters": 1000,
        }
        draft = inspect_huggingface_config(
            "example/packed",
            PINNED_REVISION,
            config,
            resident_weight_bytes_override=800,
        )
        manifest = import_huggingface_manifest(
            "example/packed",
            PINNED_REVISION,
            config,
            {"metadata": {"total_size": 700}},
            resident_weight_bytes_override=800,
        )

        self.assertTrue(draft.ready)
        self.assertEqual(manifest.model.weight_dtype, "vendor_packed")
        self.assertEqual(manifest.model.explicit_weight_bytes, 800)

    def test_missing_weight_dtype_is_unresolved_instead_of_defaulting_to_bf16(self) -> None:
        config = {
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_attention_heads": 2,
            "hidden_size": 128,
            "max_position_embeddings": 1024,
            "num_parameters": 1000,
        }

        draft = inspect_huggingface_config("example/no-dtype", PINNED_REVISION, config)

        self.assertIsNone(draft.facts["weight_dtype"])
        self.assertEqual({item.path for item in draft.unresolved}, {"weight_dtype_override"})
        with self.assertRaisesRegex(ContractError, "weight_dtype_override"):
            import_huggingface_manifest(
                "example/no-dtype",
                PINNED_REVISION,
                config,
                {"metadata": {"total_size": 1000}},
            )

    def test_partial_hybrid_config_uses_explicit_facts_and_caller_weight_format(self) -> None:
        draft = inspect_huggingface_config(
            "example/partial-hybrid",
            PINNED_REVISION,
            {
                "head_dim": 64,
                "max_position_embeddings": 4096,
                "linear_attn_config": {"head_dim": 32},
                "num_parameters": 1000,
            },
            kv_bytes_per_token_per_device_override=128,
            resident_weight_bytes_override=800,
            weight_dtype_override="vendor_packed",
        )

        self.assertEqual(draft.facts["head_dim"], 64)
        self.assertEqual(draft.facts["attention_type"], "hybrid")
        self.assertEqual(draft.facts["weight_dtype"], "vendor_packed")
        self.assertEqual(
            {item.path for item in draft.unresolved},
            {"model.num_layers", "model.num_kv_heads"},
        )

        linear_head = inspect_huggingface_config(
            "example/linear-head",
            PINNED_REVISION,
            {
                "max_position_embeddings": 4096,
                "linear_attn_config": {"head_dim": 32},
                "num_parameters": 1000,
                "torch_dtype": "float16",
            },
            kv_bytes_per_token_per_device_override=128,
        )
        self.assertEqual(linear_head.facts["head_dim"], 32)
        self.assertEqual(linear_head.field_sources["head_dim"], "config.linear_attn_config.head_dim")

    def test_model_resolution_parser_rejects_invalid_versioned_fields(self) -> None:
        draft = inspect_huggingface_config(
            "example/model",
            PINNED_REVISION,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 1024,
                "torch_dtype": "float16",
            },
        )

        missing_fact = draft.to_dict()
        del missing_fact["facts"]["architecture"]
        with self.assertRaisesRegex(ContractError, "versioned fact set"):
            ModelResolutionDraft.from_dict(missing_fact)

        bad_attention = draft.to_dict()
        bad_attention["facts"]["attention_type"] = "unknown"
        with self.assertRaisesRegex(ContractError, "attention_type is unsupported"):
            ModelResolutionDraft.from_dict(bad_attention)

        duplicate = draft.to_dict()
        duplicate["unresolved"].append(dict(duplicate["unresolved"][0]))
        with self.assertRaisesRegex(ContractError, "paths must be unique"):
            ModelResolutionDraft.from_dict(duplicate)

    def test_imports_pinned_config_and_index_metadata(self) -> None:
        config = {
            "architectures": ["ExampleForCausalLM"],
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
            "num_parameters": 7_000_000_000,
        }
        index = {"metadata": {"total_size": 14_000_000_000}, "weight_map": {"x": "model-1.safetensors"}}

        manifest = import_huggingface_manifest("example/7b", PINNED_REVISION, config, index)

        self.assertIsInstance(manifest, ModelManifest)
        self.assertEqual(manifest.model.head_dim, 128)
        self.assertIsNone(manifest.model.explicit_weight_bytes)
        self.assertEqual(manifest.artifact_bytes, 14_000_000_000)
        self.assertEqual(manifest.model.parameter_count, 7_000_000_000)
        self.assertEqual(manifest.model.weight_dtype, "bf16")
        self.assertEqual(manifest.field_sources["artifact_bytes"], "safetensors-index.metadata.total_size")

    def test_rejects_floating_revision_and_custom_attention_without_override(self) -> None:
        config = {
            "architectures": ["CustomMLAForCausalLM"],
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_attention_heads": 2,
            "hidden_size": 128,
            "max_position_embeddings": 1024,
            "torch_dtype": "bfloat16",
            "kv_lora_rank": 64,
            "num_parameters": 1000,
        }
        index = {"metadata": {"total_size": 1024}}
        with self.assertRaisesRegex(ContractError, "immutable 40-character"):
            import_huggingface_manifest("example/custom", "main", config, index)
        with self.assertRaisesRegex(ContractError, "kv_bytes_per_token_per_device_override"):
            import_huggingface_manifest("example/custom", PINNED_REVISION, config, index)

    def test_cache_round_trip_is_offline(self) -> None:
        manifest = import_huggingface_manifest(
            "example/7b",
            PINNED_REVISION,
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
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.write(path)
            restored = ModelManifest.read(path)
        self.assertEqual(restored, manifest)

    def test_resolver_fetches_pinned_metadata_once_then_uses_cache(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                    "num_parameters": 1024,
                }
            return {"metadata": {"total_size": 2048}}

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = resolve_huggingface_manifest(
                "example/7b",
                PINNED_REVISION,
                fetch_json=fetch,
                cache_dir=cache,
                inspect_safetensors_headers=False,
            )
            second = resolve_huggingface_manifest(
                "example/7b",
                PINNED_REVISION,
                fetch_json=lambda _: self.fail("cache miss"),
                cache_dir=cache,
                inspect_safetensors_headers=False,
            )
            third = resolve_huggingface_manifest(
                "example/7b",
                PINNED_REVISION,
                fetch_json=fetch,
                cache_dir=cache,
                parameter_count_override=999,
                inspect_safetensors_headers=False,
            )
        self.assertEqual(first, second)
        self.assertEqual(third.model.parameter_count, 999)
        self.assertEqual(len(calls), 4)

    def test_quantized_manifest_requires_parameter_count_provenance(self) -> None:
        config = {
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "num_attention_heads": 2,
            "hidden_size": 128,
            "max_position_embeddings": 1024,
            "torch_dtype": "float16",
            "quantization_config": {"bits": 4},
        }
        index = {"metadata": {"total_size": 512}}
        with self.assertRaisesRegex(ContractError, "parameter_count_override"):
            import_huggingface_manifest("example/quantized", PINNED_REVISION, config, index)
        manifest = import_huggingface_manifest(
            "example/quantized", PINNED_REVISION, config, index, parameter_count_override=1000
        )
        self.assertEqual(manifest.model.weight_dtype, "int4")
        self.assertEqual(manifest.model.parameter_count, 1000)

    def test_imports_nested_hybrid_quantized_model_metadata(self) -> None:
        config = {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "model_type": "kimi_k3",
            "text_config": {
                "architectures": ["KimiLinearForCausalLM"],
                "num_hidden_layers": 93,
                "num_key_value_heads": 96,
                "num_attention_heads": 96,
                "hidden_size": 7168,
                "v_head_dim": 128,
                "max_position_embeddings": 1_048_576,
                "kv_lora_rank": 512,
                "linear_attn_config": {"head_dim": 128},
                "quantization_config": {
                    "format": "mxfp4-pack-quantized",
                    "ignore": ["lm_head"],
                    "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
                },
            },
        }
        manifest = import_huggingface_manifest(
            "moonshotai/Kimi-K3",
            PINNED_REVISION,
            config,
            {"metadata": {"total_size": 1_560_860_324_864}},
            parameter_count_override=2_800_000_000_000,
            kv_bytes_per_token_per_device_override=4096,
            resident_weight_bytes_override=1_560_860_324_864,
        )

        self.assertEqual(manifest.model.num_layers, 93)
        self.assertEqual(manifest.model.num_kv_heads, 96)
        self.assertEqual(manifest.model.head_dim, 128)
        self.assertEqual(manifest.model.max_model_len, 1_048_576)
        self.assertEqual(manifest.model.attention_type, "hybrid")
        self.assertEqual(manifest.model.weight_dtype, "quantized")
        self.assertEqual(manifest.model.explicit_weight_bytes, 1_560_860_324_864)
        self.assertEqual(manifest.model.architecture, "KimiK3ForConditionalGeneration")
        self.assertEqual(manifest.field_sources["num_layers"], "config.text_config.num_hidden_layers")
        self.assertEqual(manifest.field_sources["head_dim"], "config.text_config.v_head_dim")
        self.assertIn("config.text_config.quantization_config", manifest.field_sources["weight_dtype"])

    def test_prefers_explicit_head_dim_over_hidden_size_quotient(self) -> None:
        manifest = import_huggingface_manifest(
            "example/explicit-head-dim",
            PINNED_REVISION,
            {
                "num_hidden_layers": 28,
                "num_key_value_heads": 16,
                "num_attention_heads": 16,
                "hidden_size": 3072,
                "head_dim": 256,
                "max_position_embeddings": 8192,
                "torch_dtype": "bfloat16",
                "num_parameters": 7_000_000_000,
            },
            {"metadata": {"total_size": 4_000_000_000}},
        )

        self.assertEqual(manifest.model.head_dim, 256)
        self.assertEqual(manifest.field_sources["head_dim"], "config.head_dim")

    def test_resolver_reports_all_required_kimi_inputs_before_fetching_index(self) -> None:
        config: dict[str, object] = {
            "text_config": {
                "num_hidden_layers": 93,
                "num_key_value_heads": 96,
                "num_attention_heads": 96,
                "hidden_size": 7168,
                "v_head_dim": 128,
                "max_position_embeddings": 1_048_576,
                "kv_lora_rank": 512,
                "linear_attn_config": {"head_dim": 128},
                "quantization_config": {
                    "ignore": ["lm_head"],
                    "config_groups": {"group_0": {"weights": {"num_bits": 4}}},
                },
            }
        }
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if url.endswith("config.json"):
                return config
            self.fail(f"resolver fetched metadata after detecting missing inputs: {url}")

        with self.assertRaisesRegex(
            ContractError,
            "parameter_count_override.*kv_bytes_per_token.*resident_weight_bytes_override",
        ):
            resolve_huggingface_manifest(
                "moonshotai/Kimi-K3",
                PINNED_REVISION,
                fetch_json=fetch,
                inspect_safetensors_headers=False,
            )
        self.assertEqual(len(calls), 1)

    def test_resolver_rejects_invalid_overrides_before_network_access(self) -> None:
        cases = (
            {"parameter_count_override": 0},
            {"kv_bytes_per_token_per_device_override": -1},
            {"resident_weight_bytes_override": 0},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ContractError, "positive integer"):
                    resolve_huggingface_manifest(
                        "example/model",
                        PINNED_REVISION,
                        fetch_json=lambda _: self.fail("invalid input reached the network boundary"),
                        **overrides,  # type: ignore[arg-type]
                    )

    def test_resolver_uses_pinned_api_parameter_total_for_unquantized_models(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                }
            if "/api/models/" in url:
                return {
                    "sha": PINNED_REVISION,
                    "safetensors": {"total": 1024},
                }
            return {"metadata": {"total_size": 2048}}

        manifest = resolve_huggingface_manifest(
            "example/7b",
            PINNED_REVISION,
            fetch_json=fetch,
            inspect_safetensors_headers=False,
        )

        self.assertEqual(manifest.model.parameter_count, 1024)
        self.assertEqual(manifest.field_sources["parameter_count"], "huggingface-api.safetensors.total")
        self.assertEqual(manifest.evidence[-1].metrics["parameter_count"], 1024)
        self.assertIn("/api/models/", manifest.evidence[-1].source)
        self.assertEqual(len(calls), 3)

        def mismatched(url: str) -> dict[str, object]:
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                }
            return {"sha": "b" * 40, "safetensors": {"total": 1024}}

        with self.assertRaisesRegex(ContractError, "pinned revision"):
            resolve_huggingface_manifest(
                "example/7b",
                PINNED_REVISION,
                fetch_json=mismatched,
                inspect_safetensors_headers=False,
            )

        with self.assertRaisesRegex(ContractError, "parameter_count_override"):
            resolve_huggingface_manifest(
                "example/quantized",
                PINNED_REVISION,
                fetch_json=lambda _: {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                    "quantization_config": {"bits": 4},
                },
                inspect_safetensors_headers=False,
            )

    def test_manifest_pins_tag_before_fetching_artifacts_and_caching(self) -> None:
        calls: list[str] = []

        def fetch(url: str) -> dict[str, object]:
            calls.append(url)
            if "/api/models/" in url:
                return {
                    "id": "example/model",
                    "sha": PINNED_REVISION,
                    "safetensors": {"total": 1024},
                }
            if url.endswith(f"/{PINNED_REVISION}/config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                }
            if url.endswith(f"/{PINNED_REVISION}/model.safetensors.index.json"):
                return {"metadata": {"total_size": 2048}}
            self.fail(f"unexpected metadata request: {url}")

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            manifest = resolve_huggingface_manifest(
                "example/model",
                "v1.0",
                fetch_json=fetch,
                cache_dir=cache,
                inspect_safetensors_headers=False,
            )
            cache_files = list(cache.iterdir())

        self.assertEqual(manifest.model.revision, PINNED_REVISION)
        self.assertEqual(manifest.field_sources["revision"], "huggingface-api.sha")
        self.assertEqual(manifest.evidence[-1].metrics["requested_revision"], "v1.0")
        self.assertEqual(manifest.evidence[-1].metrics["parameter_count"], 1024)
        self.assertEqual(len([url for url in calls if "/api/models/" in url]), 1)
        self.assertEqual(len(cache_files), 1)
        self.assertIn(PINNED_REVISION, cache_files[0].name)
        self.assertNotIn("v1.0", cache_files[0].name)

    def test_resolver_reads_only_safetensors_header_ranges_for_tensor_metadata(self) -> None:
        header_document = {"weight": {"dtype": "F16", "shape": [10, 20], "data_offsets": [0, 400]}}
        encoded = json.dumps(header_document, separators=(",", ":")).encode()
        file_prefix = struct.pack("<Q", len(encoded)) + encoded
        ranges: list[tuple[int, int]] = []

        def fetch_json(url: str) -> dict[str, object]:
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                    "num_parameters": 200,
                }
            return {
                "metadata": {"total_size": 400},
                "weight_map": {"weight": "model-00001-of-00001.safetensors"},
            }

        def fetch_range(url: str, start: int, end: int) -> bytes:
            self.assertTrue(url.startswith("https://huggingface.co/"))
            ranges.append((start, end))
            return file_prefix[start : end + 1]

        manifest = resolve_huggingface_manifest(
            "example/header-model",
            PINNED_REVISION,
            fetch_json=fetch_json,
            fetch_range=fetch_range,
        )
        self.assertEqual(manifest.model.parameter_count, 200)
        self.assertEqual(manifest.tensor_element_count, 200)
        self.assertEqual(ranges, [(0, 7), (8, 8 + len(encoded) - 1)])

    def test_resolver_supports_one_safetensors_file_without_an_index(self) -> None:
        header_document = {"weight": {"dtype": "F16", "shape": [10, 20], "data_offsets": [0, 400]}}
        encoded = json.dumps(header_document, separators=(",", ":")).encode()
        file_prefix = struct.pack("<Q", len(encoded)) + encoded
        ranges: list[tuple[int, int]] = []

        def fetch_json(url: str) -> dict[str, object]:
            if url.endswith("config.json"):
                return {
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 1,
                    "num_attention_heads": 2,
                    "hidden_size": 128,
                    "max_position_embeddings": 1024,
                    "torch_dtype": "float16",
                    "num_parameters": 200,
                }
            if url.endswith("model.safetensors.index.json"):
                raise FileNotFoundError
            self.fail(f"unexpected metadata request: {url}")

        def fetch_range(url: str, start: int, end: int) -> bytes:
            self.assertTrue(url.endswith("/model.safetensors"))
            ranges.append((start, end))
            return file_prefix[start : end + 1]

        manifest = resolve_huggingface_manifest(
            "example/single-file",
            PINNED_REVISION,
            fetch_json=fetch_json,
            fetch_range=fetch_range,
        )

        self.assertEqual(manifest.artifact_bytes, 400)
        self.assertEqual(manifest.tensor_element_count, 200)
        self.assertEqual(manifest.field_sources["artifact_bytes"], "model.safetensors.header.data_offsets")
        self.assertEqual(manifest.evidence[0].scope, "config.json and model.safetensors header metadata")
        self.assertEqual(ranges, [(0, 7), (8, 8 + len(encoded) - 1)])

        with self.assertRaisesRegex(ContractError, "requires header inspection"):
            resolve_huggingface_manifest(
                "example/single-file",
                PINNED_REVISION,
                fetch_json=fetch_json,
                fetch_range=fetch_range,
                inspect_safetensors_headers=False,
            )

    def test_parses_safetensors_headers_without_tensor_payloads(self) -> None:
        document = {
            "weight": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
            "scale": {"dtype": "F32", "shape": [3], "data_offsets": [12, 24]},
            "scalar": {"dtype": "F32", "shape": [], "data_offsets": [24, 28]},
            "__metadata__": {"format": "pt"},
        }
        encoded = json.dumps(document, separators=(",", ":")).encode()
        prefix = struct.pack("<Q", len(encoded)) + encoded
        self.assertEqual(parse_safetensors_header(prefix), document)
        self.assertEqual(tensor_element_count_from_safetensors_headers({"model.safetensors": prefix}), 10)
        with self.assertRaisesRegex(ContractError, "incomplete"):
            parse_safetensors_header(prefix[:-1])


class VLLMImporterTests(unittest.TestCase):
    def test_imports_initialization_memory_facts(self) -> None:
        profile = import_vllm_initialization(
            {
                **_init_identity(),
                "model_revision": PINNED_REVISION,
                "weight_bytes_per_device": 60 * GIB,
                "runtime_overhead_bytes_per_device": 2 * GIB,
                "activation_reserve_bytes_per_device": 4 * GIB,
                "kv_bytes_per_token_per_device": 4096,
                "kv_capacity_tokens_per_device": 3_000_000,
            },
            source="benchmark://init-001",
        )
        self.assertEqual(profile.weight_bytes_per_device, 60 * GIB)
        self.assertEqual(profile.evidence.kind, EvidenceKind.MEASURED)
        self.assertEqual(type(profile).from_dict(profile.to_dict()), profile)
        tampered = profile.to_dict()
        tampered["evidence"]["metrics"]["weight_bytes_per_device"] = 1
        with self.assertRaisesRegex(ContractError, "bind its exact identity"):
            type(profile).from_dict(tampered)

    def test_imports_benchmark_as_one_exact_operating_point(self) -> None:
        recipe = _recipe()
        profile = import_vllm_benchmark(
            recipe,
            {
                "completed": 100,
                "duration": 20,
                "total_input_tokens": 10_000,
                "total_output_tokens": 5_000,
                "p99_ttft_ms": 175,
                "p99_tpot_ms": 22,
            },
            context_tokens=2048,
            concurrent_sequences=16,
            latency_percentile=99,
            source="benchmark://serve-001",
        )
        self.assertEqual(profile.requests_per_second, 5)
        self.assertEqual(profile.prefill_tokens_per_second, 500)
        self.assertEqual(profile.decode_tokens_per_second, 250)
        self.assertEqual(profile.input_tokens_per_request, 100)
        self.assertEqual(profile.output_tokens_per_request, 50)
        evidence = profile.evidence[0]
        self.assertEqual(evidence.metrics["recipe_variant_fingerprint"], recipe.variant_fingerprint)
        self.assertEqual(evidence.metrics["ttft_ms"], 175)

    def test_benchmark_requires_counts_duration_and_requested_percentile(self) -> None:
        with self.assertRaisesRegex(ContractError, "completed"):
            import_vllm_benchmark(
                _recipe(),
                {"request_throughput": 5, "p99_ttft_ms": 100},
                context_tokens=2048,
                latency_percentile=99,
                source="benchmark://bad",
            )
        with self.assertRaisesRegex(ContractError, "p95_ttft_ms"):
            import_vllm_benchmark(
                _recipe(),
                {"completed": 1, "duration": 1, "total_input_tokens": 1, "total_output_tokens": 1},
                context_tokens=2048,
                latency_percentile=95,
                source="benchmark://bad",
            )

    def test_benchmark_rejects_counter_conflicts_and_impossible_context(self) -> None:
        data = {
            "completed": 10,
            "duration": 2,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
            "p99_ttft_ms": 100,
            "request_throughput": 6,
        }
        with self.assertRaisesRegex(ContractError, "conflicts"):
            import_vllm_benchmark(
                _recipe(),
                data,
                context_tokens=100,
                latency_percentile=99,
                source="benchmark://bad",
            )
        data.pop("request_throughput")
        with self.assertRaisesRegex(ContractError, "exceeds context_tokens"):
            import_vllm_benchmark(
                _recipe(),
                data,
                context_tokens=14,
                latency_percentile=99,
                source="benchmark://bad",
            )

    def test_materializes_draft_only_with_matching_manifest_and_initialization(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/7b",
                "max_num_seqs": 128,
                "max_model_len": 8192,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "kv_block_size": 16,
                "epp": {"flow_control_token_limit": 1000},
            },
            "modelservice": {
                "decode": {
                    "containers": [
                        {
                            "image": "vllm/vllm-openai:0.8.5",
                            "args": ["--kv-cache-dtype bf16 --block-size 16 --kv-events-config enabled"],
                        }
                    ]
                }
            },
        }
        draft = import_llmd_values(values, host=HardwareSpec("h100", "nvidia", 1, 80 * GIB))
        manifest = import_huggingface_manifest(
            "example/7b",
            PINNED_REVISION,
            {
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "num_attention_heads": 2,
                "hidden_size": 128,
                "max_position_embeddings": 8192,
                "torch_dtype": "float16",
                "num_parameters": 1024,
            },
            {"metadata": {"total_size": 2048}},
        )
        initialization = import_vllm_initialization(
            {
                **_init_identity(),
                "model_revision": PINNED_REVISION,
                "weight_bytes_per_device": 2048,
                "runtime_overhead_bytes_per_device": 0,
                "activation_reserve_bytes_per_device": 0,
                "kv_bytes_per_token_per_device": 512,
                "kv_capacity_tokens_per_device": ((72 * GIB - 2048) // (512 * 16)) * 16,
            },
            source="benchmark://init",
        )
        recipe = materialize_recipe_draft(draft, manifest, initialization, precise_prefix_routing=True)
        self.assertEqual(recipe.model.revision, PINNED_REVISION)
        self.assertEqual(recipe.runtime.weight_bytes_per_device_override, 2048)

        other_manifest = replace(manifest, model=replace(manifest.model, model_id="example/other"))
        with self.assertRaisesRegex(ContractError, "identity does not match"):
            materialize_recipe_draft(draft, other_manifest, initialization, precise_prefix_routing=True)

        shorter_manifest = replace(manifest, model=replace(manifest.model, max_model_len=4096))
        with self.assertRaisesRegex(ContractError, "max_model_len exceeds"):
            materialize_recipe_draft(draft, shorter_manifest, initialization, precise_prefix_routing=True)
        self.assertEqual(len(recipe.evidence), 3)

        mismatched = import_vllm_initialization(
            {
                **_init_identity(),
                "model_revision": "b" * 40,
                "weight_bytes_per_device": 2048,
                "runtime_overhead_bytes_per_device": 0,
                "activation_reserve_bytes_per_device": 0,
                "kv_bytes_per_token_per_device": 512,
                "kv_capacity_tokens_per_device": ((72 * GIB - 2048) // (512 * 16)) * 16,
            },
            source="benchmark://init",
        )
        with self.assertRaisesRegex(ContractError, "does not match"):
            materialize_recipe_draft(draft, manifest, mismatched, precise_prefix_routing=True)
        runtime_mismatched = import_vllm_initialization(
            {
                **_init_identity(runtime_version="different-runtime"),
                "model_revision": PINNED_REVISION,
                "weight_bytes_per_device": 2048,
                "runtime_overhead_bytes_per_device": 0,
                "activation_reserve_bytes_per_device": 0,
                "kv_bytes_per_token_per_device": 512,
                "kv_capacity_tokens_per_device": ((72 * GIB - 2048) // (512 * 16)) * 16,
            },
            source="benchmark://init",
        )
        with self.assertRaisesRegex(ContractError, "runtime_version does not match"):
            materialize_recipe_draft(draft, manifest, runtime_mismatched, precise_prefix_routing=True)

        inconsistent_capacity = import_vllm_initialization(
            {
                **_init_identity(),
                "model_revision": PINNED_REVISION,
                "weight_bytes_per_device": 2048,
                "runtime_overhead_bytes_per_device": 0,
                "activation_reserve_bytes_per_device": 0,
                "kv_bytes_per_token_per_device": 512,
                "kv_capacity_tokens_per_device": 1,
            },
            source="benchmark://init",
        )
        with self.assertRaisesRegex(ContractError, "does not reconcile"):
            materialize_recipe_draft(draft, manifest, inconsistent_capacity, precise_prefix_routing=True)


if __name__ == "__main__":
    unittest.main()
