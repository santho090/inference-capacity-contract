"""Small JSON CLI for the public contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters import to_llmd_planner_payload, to_scaling_policy_input
from .calculator import capacity_for, what_fits
from .models import CapacityContract, HardwareInventory, HardwareSpec, ModelSpec, RuntimeVariant, WorkloadProfile
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
        else:
            contract = CapacityContract.from_dict(_read(args.contract))
            profile = WorkloadProfile.from_dict(_read(args.profile))
            _write(recommend_scale(contract, profile).to_dict(), args.output)
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"icc: error: {exc}", file=sys.stderr)
        return 2
