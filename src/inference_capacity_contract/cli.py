"""Small JSON CLI for the public contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import to_llmd_planner_payload, to_scaling_policy_input
from .calculator import capacity_for, what_fits
from .drafts import RecipeDraft, audit_recipe_draft, import_llmd_values
from .importers import (
    ModelManifest,
    VLLMInitializationProfile,
    import_huggingface_manifest,
    import_vllm_benchmark,
    import_vllm_initialization,
    materialize_recipe_draft,
    resolve_huggingface_manifest,
)
from .inventory import MeasurementInventory, ProviderInventory, RuntimeInventory
from .models import CapacityContract, HardwareInventory, HardwareSpec, ModelSpec, RuntimeVariant, WorkloadProfile
from .planner import PlanningObjective, explore
from .planner import plan as plan_providers
from .recipe import LoadRequirement, ServingRecipe, audit_recipe
from .scaling import recommend_scale


def _read(path: str) -> dict[str, Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(value: Any, output: str | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def _planner_model(model_path: str | None, manifest_path: str | None) -> ModelSpec | ModelManifest:
    if model_path is not None:
        return ModelSpec.from_dict(_read(model_path))
    if manifest_path is not None:
        return ModelManifest.from_dict(_read(manifest_path))
    raise ValueError("one of --model or --manifest is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="icc", description="Compute deterministic LLM capacity contracts.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="calculate one model/hardware/runtime contract")
    plan.add_argument("--model", required=True, help="model JSON file or -")
    plan.add_argument("--hardware", required=True, help="hardware JSON file or -")
    plan.add_argument("--runtime", required=True, help="runtime JSON file or -")
    plan.add_argument("--output")

    fit = sub.add_parser("fit", help="find which inventory entries fit a model")
    fit.add_argument("--model", required=True)
    fit.add_argument("--inventory", required=True)
    fit.add_argument("--runtime", required=True, help="runtime JSON file")
    fit.add_argument("--output")

    validate = sub.add_parser("validate", help="round-trip a contract JSON document")
    validate.add_argument("--contract", required=True)
    validate.add_argument("--output")

    export = sub.add_parser("export", help="export a contract to an upstream adapter payload")
    export.add_argument("--contract", required=True)
    export.add_argument("--target", choices=("llmd-planner", "scaling-policy"), required=True)
    export.add_argument("--output")

    scale = sub.add_parser("scale", help="recommend replicas from a measured workload profile")
    scale.add_argument("--contract", required=True)
    scale.add_argument("--profile", required=True)
    scale.add_argument("--output")

    audit = sub.add_parser("audit", help="audit a normalized serving recipe against a requested load")
    audit.add_argument("--recipe", required=True)
    audit.add_argument("--load", required=True)
    audit.add_argument("--output")

    import_llmd = sub.add_parser("import-llmd", help="normalize caller-parsed llm-d values into a recipe draft")
    import_llmd.add_argument("--values", required=True, help="llm-d values JSON file")
    import_llmd.add_argument("--hardware", required=True, help="hardware JSON file")
    import_llmd.add_argument("--recipe-id")
    import_llmd.add_argument("--source", default="llmd-values://caller-supplied")
    import_llmd.add_argument("--output")

    audit_draft = sub.add_parser("audit-draft", help="audit topology and routing facts in a recipe draft")
    audit_draft.add_argument("--draft", required=True)
    audit_draft.add_argument("--output")

    manifest = sub.add_parser("import-model-manifest", help="normalize pinned Hugging Face metadata")
    manifest.add_argument("--repo-id", required=True)
    manifest.add_argument("--revision", required=True)
    manifest.add_argument("--config", required=True)
    manifest.add_argument("--safetensors-index", required=True)
    manifest.add_argument("--kv-bytes-per-token-per-device", type=int)
    manifest.add_argument("--parameter-count", type=int)
    manifest.add_argument("--output")

    resolve_model = sub.add_parser(
        "resolve-model",
        help="resolve a pinned public Hugging Face model into a replayable manifest",
    )
    resolve_model.add_argument("--repo-id", required=True)
    resolve_model.add_argument("--revision", required=True, help="immutable 40-character commit SHA")
    resolve_model.add_argument("--cache-dir")
    resolve_model.add_argument("--kv-bytes-per-token-per-device", type=int)
    resolve_model.add_argument("--parameter-count", type=int)
    resolve_model.add_argument("--resident-weight-bytes", type=int)
    resolve_model.add_argument(
        "--inspect-safetensors-headers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    resolve_model.add_argument("--output")

    initialization = sub.add_parser("import-vllm-init", help="normalize vLLM initialization memory evidence")
    initialization.add_argument("--input", required=True)
    initialization.add_argument("--source", required=True)
    initialization.add_argument("--output")

    materialize = sub.add_parser(
        "materialize-recipe",
        help="complete a recipe draft with a pinned model manifest and initialization profile",
    )
    materialize.add_argument("--draft", required=True)
    materialize.add_argument("--manifest", required=True)
    materialize.add_argument("--initialization", required=True)
    materialize.add_argument(
        "--precise-prefix-routing",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="declare whether the deployed routing policy uses precise prefix routing",
    )
    materialize.add_argument("--output")

    benchmark = sub.add_parser("import-vllm-benchmark", help="bind vLLM benchmark counters to a recipe")
    benchmark.add_argument("--recipe", required=True)
    benchmark.add_argument("--benchmark", required=True)
    benchmark.add_argument("--context-tokens", required=True, type=int)
    benchmark.add_argument("--concurrent-sequences", type=int)
    benchmark.add_argument("--latency-percentile", required=True, type=float)
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--output")

    explore_parser = sub.add_parser(
        "explore",
        help="compare one model across provider instances and runtime options",
    )
    explore_model = explore_parser.add_mutually_exclusive_group(required=True)
    explore_model.add_argument("--model")
    explore_model.add_argument("--manifest")
    explore_parser.add_argument("--providers", required=True)
    explore_parser.add_argument("--runtimes", required=True)
    explore_parser.add_argument("--context-tokens", required=True, type=int)
    explore_parser.add_argument("--output")

    provider_plan = sub.add_parser(
        "plan-providers",
        help="size one model and load across provider instances and runtime options",
    )
    plan_model = provider_plan.add_mutually_exclusive_group(required=True)
    plan_model.add_argument("--model")
    plan_model.add_argument("--manifest")
    provider_plan.add_argument("--providers", required=True)
    provider_plan.add_argument("--runtimes", required=True)
    provider_plan.add_argument("--load", required=True)
    provider_plan.add_argument("--measurements")
    provider_plan.add_argument(
        "--objective",
        choices=tuple(item.value for item in PlanningObjective),
        default=PlanningObjective.LOWEST_COST.value,
    )
    provider_plan.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = capacity_for(
                ModelSpec.from_dict(_read(args.model)),
                HardwareSpec.from_dict(_read(args.hardware)),
                RuntimeVariant.from_dict(_read(args.runtime)),
            )
            _write(result.to_dict(), args.output)
        elif args.command == "fit":
            model = ModelSpec.from_dict(_read(args.model))
            inventory = HardwareInventory.from_dict(_read(args.inventory))
            runtime = RuntimeVariant.from_dict(_read(args.runtime))
            _write(
                {
                    "schema_version": "capacity-contract-fit-2.0",
                    "candidates": [item.to_dict() for item in what_fits(model, inventory, runtime)],
                },
                args.output,
            )
        elif args.command == "validate":
            contract = CapacityContract.from_dict(_read(args.contract))
            _write(contract.to_dict(), args.output)
        elif args.command == "export":
            contract = CapacityContract.from_dict(_read(args.contract))
            exporter = to_llmd_planner_payload if args.target == "llmd-planner" else to_scaling_policy_input
            _write(exporter(contract), args.output)
        elif args.command == "scale":
            contract = CapacityContract.from_dict(_read(args.contract))
            profile = WorkloadProfile.from_dict(_read(args.profile))
            _write(recommend_scale(contract, profile).to_dict(), args.output)
        elif args.command == "audit":
            recipe = ServingRecipe.from_dict(_read(args.recipe))
            load = LoadRequirement.from_dict(_read(args.load))
            _write(audit_recipe(recipe, load).to_dict(), args.output)
        elif args.command == "import-llmd":
            draft = import_llmd_values(
                _read(args.values),
                host=HardwareSpec.from_dict(_read(args.hardware)),
                recipe_id=args.recipe_id,
                source=args.source,
            )
            _write(draft.to_dict(), args.output)
        elif args.command == "audit-draft":
            _write(audit_recipe_draft(RecipeDraft.from_dict(_read(args.draft))).to_dict(), args.output)
        elif args.command == "import-model-manifest":
            manifest = import_huggingface_manifest(
                args.repo_id,
                args.revision,
                _read(args.config),
                _read(args.safetensors_index),
                kv_bytes_per_token_per_device_override=args.kv_bytes_per_token_per_device,
                parameter_count_override=args.parameter_count,
            )
            _write(manifest.to_dict(), args.output)
        elif args.command == "resolve-model":
            resolved = resolve_huggingface_manifest(
                args.repo_id,
                args.revision,
                cache_dir=None if args.cache_dir is None else Path(args.cache_dir),
                kv_bytes_per_token_per_device_override=args.kv_bytes_per_token_per_device,
                parameter_count_override=args.parameter_count,
                resident_weight_bytes_override=args.resident_weight_bytes,
                inspect_safetensors_headers=args.inspect_safetensors_headers,
            )
            _write(resolved.to_dict(), args.output)
        elif args.command == "import-vllm-init":
            _write(import_vllm_initialization(_read(args.input), source=args.source).to_dict(), args.output)
        elif args.command == "materialize-recipe":
            recipe = materialize_recipe_draft(
                RecipeDraft.from_dict(_read(args.draft)),
                ModelManifest.from_dict(_read(args.manifest)),
                VLLMInitializationProfile.from_dict(_read(args.initialization)),
                precise_prefix_routing=args.precise_prefix_routing,
            )
            _write(recipe.to_dict(), args.output)
        elif args.command == "import-vllm-benchmark":
            recipe = ServingRecipe.from_dict(_read(args.recipe))
            benchmark_profile = import_vllm_benchmark(
                recipe,
                _read(args.benchmark),
                context_tokens=args.context_tokens,
                concurrent_sequences=args.concurrent_sequences,
                latency_percentile=args.latency_percentile,
                source=args.source,
            )
            _write(benchmark_profile.to_dict(), args.output)
        elif args.command == "explore":
            exploration = explore(
                _planner_model(args.model, args.manifest),
                ProviderInventory.from_dict(_read(args.providers)),
                RuntimeInventory.from_dict(_read(args.runtimes)),
                context_tokens=args.context_tokens,
            )
            _write(exploration.to_dict(), args.output)
        elif args.command == "plan-providers":
            capacity_plan = plan_providers(
                _planner_model(args.model, args.manifest),
                ProviderInventory.from_dict(_read(args.providers)),
                RuntimeInventory.from_dict(_read(args.runtimes)),
                LoadRequirement.from_dict(_read(args.load)),
                objective=PlanningObjective(args.objective),
                measurements=(
                    None if args.measurements is None else MeasurementInventory.from_dict(_read(args.measurements))
                ),
            )
            _write(capacity_plan.to_dict(), args.output)
        else:
            raise ValueError(f"unknown command {args.command!r}")
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"icc: error: {exc}", file=sys.stderr)
        return 2
