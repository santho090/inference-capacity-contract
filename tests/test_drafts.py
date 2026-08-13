from __future__ import annotations

import json
import unittest
from pathlib import Path

from inference_capacity_contract import (
    ContractError,
    HardwareSpec,
    RecipeDraft,
    StructuralAuditStatus,
    audit_recipe_draft,
    import_llmd_values,
)

GIB = 1024**3
ROOT = Path(__file__).parents[1]


def _container_args(*lines: str) -> list[dict[str, object]]:
    return [{"name": "vllm", "image": "vllm/vllm-openai:v0.25.0", "args": [" ".join(lines)]}]


class RecipeDraftTests(unittest.TestCase):
    def test_sanitized_golden_shapes_remain_stable(self) -> None:
        cases = (
            ("llmd-values-long-context-tp8.json", "hardware-accelerator-x8-288gb.json", 8, 0, 2),
            ("llmd-values-dp4-on-x8.json", "hardware-accelerator-x8-141gb.json", 4, 4, 4),
            ("llmd-values-dp8-nvidia.json", "hardware-accelerator-x8-288gb.json", 8, 0, 8),
            ("llmd-values-dp8-amd.json", "hardware-amd-accelerator-x8-288gb.json", 8, 0, 4),
        )
        for values_name, hardware_name, ranks, unused, full_context in cases:
            with self.subTest(values=values_name):
                values = json.loads((ROOT / "docs" / "fixtures" / values_name).read_text(encoding="utf-8"))
                hardware = HardwareSpec.from_dict(
                    json.loads((ROOT / "docs" / "fixtures" / hardware_name).read_text(encoding="utf-8"))
                )
                audit = audit_recipe_draft(import_llmd_values(values, host=hardware))
                self.assertEqual(audit.engine_rank_count, ranks)
                self.assertEqual(audit.unused_device_count, unused)
                self.assertEqual(audit.flow_control_full_context_sequences, full_context)

    def test_imports_known_shape_and_reports_missing_memory_facts(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/custom-moe",
                "max_num_seqs": 24,
                "max_model_len": 1_048_576,
                "tensor_parallel_size": 8,
                "data_parallel_size": 1,
                "kv_block_size": 1536,
                "epp": {
                    "flow_control_token_limit": 2_104_030,
                    "output_ratio": 0.2,
                    "max_estimated_output_tokens": 500,
                },
            },
            "modelservice": {
                "modelArtifacts": {"uri": "hf://example/custom-moe", "name": "example/custom-moe"},
                "decode": {
                    "containers": _container_args(
                        "vllm serve example/custom-moe",
                        "--tensor-parallel-size 8",
                        "--max-model-len 1048576",
                        "--max-num-seqs 24",
                        "--block-size 1536",
                        "--kv-cache-dtype fp8",
                        "--gpu-memory-utilization 0.96",
                        "--enable-expert-parallel",
                        "--kv-events-config enabled",
                    )
                },
            },
        }
        host = HardwareSpec("accelerator-x8", "nvidia", 8, 288 * GIB, 0.90)

        draft = import_llmd_values(values, host=host, recipe_id="custom-moe-x8")

        self.assertFalse(draft.ready)
        self.assertEqual(draft.document["host"]["memory_utilization_limit"], 0.96)
        self.assertEqual(draft.document["topology"]["expert_parallel_size"], 8)
        self.assertIn("model.revision", draft.unresolved_paths)
        self.assertIn("runtime.weight_bytes_per_device_override", draft.unresolved_paths)
        self.assertIn("model.kv_bytes_per_token_per_device_override", draft.unresolved_paths)
        self.assertIn("llmd.precise_prefix_routing", draft.unresolved_paths)
        self.assertNotIn("model.model_id", draft.unresolved_paths)
        self.assertNotIn("host.hardware_id", draft.unresolved_paths)
        self.assertNotIn("runtime.engine", draft.unresolved_paths)
        self.assertNotIn("runtime.tensor_parallel_size", draft.unresolved_paths)
        self.assertNotIn("topology.physical_device_count", draft.unresolved_paths)
        serialized_paths = {item["path"] for item in draft.to_dict()["unresolved"]}
        self.assertEqual(serialized_paths, set(draft.unresolved_paths))
        self.assertEqual(RecipeDraft.from_dict(draft.to_dict()), draft)
        tampered = draft.to_dict()
        tampered["ready"] = True
        with self.assertRaisesRegex(ContractError, "ready flag"):
            RecipeDraft.from_dict(tampered)
        with self.assertRaisesRegex(ContractError, "cannot build a serving recipe"):
            draft.to_serving_recipe()

    def test_structural_audit_exposes_long_context_and_block_alignment(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/custom-moe",
                "max_num_seqs": 24,
                "max_model_len": 1_048_576,
                "tensor_parallel_size": 8,
                "data_parallel_size": 1,
                "kv_block_size": 1536,
                "epp": {"flow_control_token_limit": 2_104_030},
            },
            "modelservice": {"decode": {"containers": _container_args("--enable-expert-parallel")}},
        }
        draft = import_llmd_values(values, host=HardwareSpec("x8", "nvidia", 8, 288 * GIB))

        audit = audit_recipe_draft(draft)

        self.assertEqual(audit.status, StructuralAuditStatus.INCOMPLETE)
        self.assertEqual(audit.engine_rank_count, 8)
        self.assertEqual(audit.unused_device_count, 0)
        self.assertEqual(audit.configured_max_concurrent_sequences, 24)
        self.assertEqual(audit.flow_control_full_context_sequences, 2)
        self.assertTrue(any("not divisible" in warning for warning in audit.warnings))

    def test_flow_control_full_context_capacity_is_block_aligned(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/boundary",
                "max_num_seqs": 2,
                "max_model_len": 600,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "kv_block_size": 512,
                "epp": {"flow_control_token_limit": 1000},
            },
            "modelservice": {"decode": {"containers": _container_args("--block-size 512")}},
        }
        audit = audit_recipe_draft(import_llmd_values(values, host=HardwareSpec("accelerator", "nvidia", 1, 80 * GIB)))

        self.assertEqual(audit.flow_control_full_context_sequences, 0)
        self.assertIn(
            "flow-control budget is smaller than one configured maximum-context sequence",
            audit.warnings,
        )

    def test_draft_parser_restores_omitted_required_facts(self) -> None:
        draft = import_llmd_values({}, host=None)
        document = draft.to_dict()
        document["unresolved"] = []
        document["ready"] = True

        with self.assertRaisesRegex(ContractError, "ready flag"):
            RecipeDraft.from_dict(document)

    def test_structural_audit_flags_unassigned_devices(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/custom-moe",
                "max_num_seqs": 128,
                "max_model_len": 262_144,
                "tensor_parallel_size": 1,
                "data_parallel_size": 4,
                "epp": {"flow_control_token_limit": 1_100_000},
            },
            "modelservice": {"decode": {"containers": _container_args("--enable-expert-parallel", "--block-size 256")}},
        }
        draft = import_llmd_values(values, host=HardwareSpec("accelerator-x8", "nvidia", 8, 141 * GIB))

        audit = audit_recipe_draft(draft)

        self.assertEqual(audit.engine_rank_count, 4)
        self.assertEqual(audit.unused_device_count, 4)
        self.assertEqual(audit.configured_max_concurrent_sequences, 512)
        self.assertTrue(any("4 physical devices" in warning for warning in audit.warnings))

    def test_structural_audit_rejects_more_engine_ranks_than_devices(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/model",
                "max_num_seqs": 8,
                "max_model_len": 4096,
                "tensor_parallel_size": 2,
                "data_parallel_size": 8,
            },
            "modelservice": {"decode": {"containers": _container_args("--block-size 16")}},
        }
        draft = import_llmd_values(values, host=HardwareSpec("accelerator-x8", "nvidia", 8, 80 * GIB))

        audit = audit_recipe_draft(draft)

        self.assertEqual(audit.status, StructuralAuditStatus.INVALID)
        self.assertTrue(any("exceeds" in issue for issue in audit.issues))

    def test_structural_audit_rejects_runtime_llmd_block_mismatch(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/model",
                "max_num_seqs": 8,
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "kv_block_size": 32,
            },
            "modelservice": {"decode": {"containers": _container_args("--block-size 16")}},
        }
        audit = audit_recipe_draft(import_llmd_values(values, host=HardwareSpec("accelerator", "nvidia", 1, 80 * GIB)))
        self.assertEqual(audit.status, StructuralAuditStatus.INVALID)
        self.assertIn("llm-d and runtime KV block sizes do not match", audit.issues)

    def test_missing_parallelism_stays_unresolved_instead_of_defaulting(self) -> None:
        values = {
            "metadata": {"model_name": "example/model", "max_num_seqs": 8, "max_model_len": 4096},
            "modelservice": {"decode": {"containers": _container_args("--block-size 16")}},
        }
        draft = import_llmd_values(values, host=HardwareSpec("accelerator", "nvidia", 1, 80 * GIB))
        audit = audit_recipe_draft(draft)
        self.assertIsNone(audit.engine_rank_count)
        self.assertIn("topology.tensor_parallel_size", draft.unresolved_paths)
        self.assertIn("topology.data_parallel_size", draft.unresolved_paths)

    def test_imports_matching_immutable_model_revision_and_rejects_conflicts(self) -> None:
        revision = "a" * 40
        values = {
            "metadata": {"model_name": "example/model", "model_revision": revision},
            "modelservice": {
                "modelArtifacts": {"uri": f"hf://example/model@{revision}"},
                "decode": {"containers": _container_args(f"--revision {revision}")},
            },
        }

        draft = import_llmd_values(values)

        self.assertEqual(draft.document["model"]["model_id"], "example/model")
        self.assertEqual(draft.document["model"]["revision"], revision)
        self.assertNotIn("model.revision", draft.unresolved_paths)

        conflict_values = {
            **values,
            "metadata": {"model_name": "example/model", "model_revision": "b" * 40},
        }
        with self.assertRaisesRegex(ContractError, "conflicting immutable model revisions"):
            import_llmd_values(conflict_values)

    def test_standard_attention_and_non_ep_do_not_require_runtime_overrides(self) -> None:
        values = {
            "metadata": {
                "model_name": "example/model",
                "max_num_seqs": 8,
                "max_model_len": 4096,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            },
            "modelservice": {
                "decode": {
                    "containers": [
                        {"name": "metrics-sidecar", "image": "example/metrics:1", "args": []},
                        {
                            "name": "vllm",
                            "image": "vllm/vllm-openai:0.8.5",
                            "args": ["--block-size 16 --kv-cache-dtype bf16"],
                        },
                    ]
                }
            },
        }
        draft = import_llmd_values(values, host=HardwareSpec("accelerator", "nvidia", 1, 80 * GIB))
        completed = draft.with_overrides(
            {
                "model": {
                    "revision": "a" * 40,
                    "parameter_count": 1,
                    "num_layers": 1,
                    "num_kv_heads": 1,
                    "head_dim": 1,
                    "weight_dtype": "bf16",
                    "attention_type": "mha",
                },
                "runtime": {"runtime_overhead_bytes_per_device": 0, "activation_reserve_bytes_per_device": 0},
            }
        )
        self.assertEqual(completed.document["runtime"]["version"], "0.8.5")
        self.assertNotIn("model.kv_bytes_per_token_per_device_override", completed.unresolved_paths)
        self.assertNotIn("runtime.weight_bytes_per_device_override", completed.unresolved_paths)


if __name__ == "__main__":
    unittest.main()
