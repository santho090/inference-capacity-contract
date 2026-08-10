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


if __name__ == "__main__":
    unittest.main()
