"""Read the JSON Schemas shipped with the library."""

from __future__ import annotations

import json
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, cast

from .models import ContractError


def _schema_dir() -> Traversable:
    bundled = files("inference_capacity_contract").joinpath("schemas")
    if bundled.is_dir():
        return bundled
    return Path(__file__).parents[2] / "schemas"


def available_schema_versions() -> tuple[str, ...]:
    """Return the schema versions bundled in this installation."""

    suffix = ".schema.json"
    return tuple(
        sorted(item.name.removesuffix(suffix) for item in _schema_dir().iterdir() if item.name.endswith(suffix))
    )


def load_schema(schema_version: str) -> dict[str, Any]:
    """Load one bundled JSON Schema by its document schema version."""

    if schema_version not in available_schema_versions():
        raise ContractError(f"unknown schema version {schema_version!r}")
    value = json.loads(_schema_dir().joinpath(f"{schema_version}.schema.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"schema {schema_version!r} is not a JSON object")
    return cast(dict[str, Any], value)
