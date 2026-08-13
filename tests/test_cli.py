import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class CliTests(unittest.TestCase):
    def test_plan_validate_and_export(self) -> None:
        gib = 1024**3
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model = directory_path / "model.json"
            hardware = directory_path / "hardware.json"
            runtime = directory_path / "runtime.json"
            contract = directory_path / "contract.json"
            profile = directory_path / "profile.json"
            model.write_text(
                json.dumps(
                    {
                        "model_id": "example/7b",
                        "revision": "r1",
                        "parameter_count": 7_000_000_000,
                        "num_layers": 32,
                        "num_kv_heads": 8,
                        "head_dim": 128,
                        "max_model_len": 4096,
                    }
                ),
                encoding="utf-8",
            )
            hardware.write_text(
                json.dumps(
                    {
                        "hardware_id": "h100",
                        "vendor": "nvidia",
                        "device_count": 1,
                        "memory_bytes_per_device": 80 * gib,
                    }
                ),
                encoding="utf-8",
            )
            runtime.write_text(json.dumps({"engine": "vllm", "version": "0.8.5"}), encoding="utf-8")

            command = [
                sys.executable,
                "-m",
                "inference_capacity_contract",
                "plan",
                "--model",
                str(model),
                "--hardware",
                str(hardware),
                "--runtime",
                str(runtime),
                "--output",
                str(contract),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(contract.read_text(encoding="utf-8"))["fits"])
            profile.write_text(
                json.dumps(
                    {
                        "profile_id": "profile-001",
                        "model_id": "example/7b",
                        "model_revision": "r1",
                        "hardware_id": "h100",
                        "runtime_engine": "vllm",
                        "runtime_version": "0.8.5",
                        "request_rate_per_second": 5,
                        "input_tokens_per_request": 100,
                        "output_tokens_per_request": 50,
                        "sustainable_requests_per_replica_per_second": 5,
                        "sustainable_prefill_tokens_per_replica_per_second": 1000,
                        "sustainable_decode_tokens_per_replica_per_second": 500,
                        "evidence": [
                            {
                                "evidence_id": "run-001",
                                "kind": "measured",
                                "source": "benchmark://run-001",
                                "scope": "example/7b@r1 on h100/vllm-0.8.5",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "inference_capacity_contract", "validate", "--contract", str(contract)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            exported = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "export",
                    "--contract",
                    str(contract),
                    "--target",
                    "scaling-policy",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(exported.returncode, 0, exported.stderr)
            self.assertIsNone(json.loads(exported.stdout)["replica_count"])
            scaled = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "scale",
                    "--contract",
                    str(contract),
                    "--profile",
                    str(profile),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(scaled.returncode, 0, scaled.stderr)
            self.assertGreaterEqual(json.loads(scaled.stdout)["recommended_replicas"], 1)

            audited = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "audit",
                    "--recipe",
                    str(ROOT / "docs" / "fixtures" / "serving-recipe-hybrid-tp8.json"),
                    "--load",
                    str(ROOT / "docs" / "fixtures" / "load-context-concurrency.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audited.returncode, 0, audited.stderr)
            audit_document = json.loads(audited.stdout)
            self.assertEqual(audit_document["status"], "sufficient")
            self.assertEqual(audit_document["required_devices"], 16)

            draft = directory_path / "draft.json"
            imported = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "import-llmd",
                    "--values",
                    str(ROOT / "docs" / "fixtures" / "llmd-values-long-context-tp8.json"),
                    "--hardware",
                    str(ROOT / "docs" / "fixtures" / "hardware-accelerator-x8-288gb.json"),
                    "--output",
                    str(draft),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertFalse(json.loads(draft.read_text(encoding="utf-8"))["ready"])
            structural = subprocess.run(
                [sys.executable, "-m", "inference_capacity_contract", "audit-draft", "--draft", str(draft)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(structural.returncode, 0, structural.stderr)
            structural_document = json.loads(structural.stdout)
            self.assertEqual(structural_document["flow_control_full_context_sequences"], 2)

            benchmark_import = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "import-vllm-benchmark",
                    "--recipe",
                    str(ROOT / "docs" / "fixtures" / "serving-recipe-hybrid-tp8.json"),
                    "--benchmark",
                    str(ROOT / "docs" / "fixtures" / "vllm-benchmark-serve.json"),
                    "--context-tokens",
                    "1048576",
                    "--concurrent-sequences",
                    "16",
                    "--latency-percentile",
                    "99",
                    "--source",
                    "benchmark://cli-fixture",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(benchmark_import.returncode, 0, benchmark_import.stderr)
            self.assertEqual(json.loads(benchmark_import.stdout)["requests_per_second"], 5)

            initialization_import = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "import-vllm-init",
                    "--input",
                    str(ROOT / "docs" / "fixtures" / "vllm-initialization.json"),
                    "--source",
                    "benchmark://cli-initialization",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(initialization_import.returncode, 0, initialization_import.stderr)
            self.assertEqual(
                json.loads(initialization_import.stdout)["schema_version"],
                "vllm-initialization-profile-1.0",
            )

            config = directory_path / "config.json"
            index = directory_path / "model.safetensors.index.json"
            config.write_text(
                json.dumps(
                    {
                        "num_hidden_layers": 2,
                        "num_key_value_heads": 1,
                        "num_attention_heads": 2,
                        "hidden_size": 128,
                        "max_position_embeddings": 1024,
                        "torch_dtype": "float16",
                        "num_parameters": 1024,
                    }
                ),
                encoding="utf-8",
            )
            index.write_text(json.dumps({"metadata": {"total_size": 2048}}), encoding="utf-8")
            manifest_import = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "import-model-manifest",
                    "--repo-id",
                    "example/model",
                    "--revision",
                    "a" * 40,
                    "--config",
                    str(config),
                    "--safetensors-index",
                    str(index),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(manifest_import.returncode, 0, manifest_import.stderr)
            self.assertEqual(json.loads(manifest_import.stdout)["artifact_bytes"], 2048)

    def test_llmd_import_materialization_and_audit_workflow(self) -> None:
        fixtures = ROOT / "docs" / "fixtures"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            draft = output / "draft.json"
            manifest = output / "manifest.json"
            initialization = output / "initialization.json"
            recipe = output / "recipe.json"

            commands = (
                [
                    "import-llmd",
                    "--values",
                    str(fixtures / "llmd-values-long-context-tp8.json"),
                    "--hardware",
                    str(fixtures / "hardware-accelerator-x8-288gb.json"),
                    "--output",
                    str(draft),
                ],
                [
                    "import-model-manifest",
                    "--repo-id",
                    "example/long-context-moe",
                    "--revision",
                    "a" * 40,
                    "--config",
                    str(fixtures / "model-config-long-context-moe.json"),
                    "--safetensors-index",
                    str(fixtures / "model-safetensors-index-long-context-moe.json"),
                    "--kv-bytes-per-token-per-device",
                    "4096",
                    "--output",
                    str(manifest),
                ],
                [
                    "import-vllm-init",
                    "--input",
                    str(fixtures / "vllm-initialization.json"),
                    "--source",
                    "benchmark://cli-initialization",
                    "--output",
                    str(initialization),
                ],
                [
                    "materialize-recipe",
                    "--draft",
                    str(draft),
                    "--manifest",
                    str(manifest),
                    "--initialization",
                    str(initialization),
                    "--precise-prefix-routing",
                    "--output",
                    str(recipe),
                ],
            )
            for command in commands:
                result = subprocess.run(
                    [sys.executable, "-m", "inference_capacity_contract", *command],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            recipe_document = json.loads(recipe.read_text(encoding="utf-8"))
            self.assertEqual(recipe_document["model"]["model_id"], "example/long-context-moe")
            self.assertEqual(recipe_document["model"]["max_model_len"], 1_048_576)
            self.assertEqual(recipe_document["runtime"]["weight_bytes_per_device_override"], 60 * 1024**3)

            audit = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "inference_capacity_contract",
                    "audit",
                    "--recipe",
                    str(recipe),
                    "--load",
                    str(fixtures / "load-context-concurrency.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(audit.returncode, 0, audit.stderr)
            audit_document = json.loads(audit.stdout)
            self.assertEqual(audit_document["status"], "insufficient")
            self.assertEqual(audit_document["required_groups"], 2)


if __name__ == "__main__":
    unittest.main()
