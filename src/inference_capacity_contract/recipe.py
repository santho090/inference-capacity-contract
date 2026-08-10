"""Audit normalized serving recipes against model, memory, and load constraints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from math import isclose, isfinite
from typing import Any

from .calculator import capacity_for
from .models import (
    CapacityContract,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareSpec,
    ModelSpec,
    RuntimeVariant,
)

RECIPE_VARIANT_FINGERPRINT_VERSION = "serving-recipe-variant-1.0"


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _non_negative_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0:
        raise ContractError(f"{name} must be finite and non-negative")
    return normalized


def _optional_positive_int(name: str, value: Any) -> int | None:
    return None if value is None else _positive_int(name, value)


def _optional_positive_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    normalized = _non_negative_float(name, value)
    if normalized == 0:
        raise ContractError(f"{name} must be positive")
    return normalized


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _sequence(name: str, value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be an array")
    return tuple(value)


def _required(data: Mapping[str, Any], name: str) -> Any:
    if name not in data:
        raise ContractError(f"missing required field {name!r}")
    return data[name]


def _required_str(data: Mapping[str, Any], name: str) -> str:
    value = _required(data, name)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ParallelTopology:
    """Engine ranks and physical devices used by one serving group."""

    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    physical_device_count: int = 1
    independent_kv_ranks: int = 1

    def __post_init__(self) -> None:
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
            "physical_device_count",
            "independent_kv_ranks",
        ):
            _positive_int(name, getattr(self, name))
        if self.engine_rank_count > self.physical_device_count:
            raise ContractError("TP x DP cannot exceed physical_device_count")
        if self.engine_rank_count % self.expert_parallel_size != 0:
            raise ContractError("expert_parallel_size must divide TP x DP")
        if self.independent_kv_ranks > self.data_parallel_size:
            raise ContractError("independent_kv_ranks cannot exceed data_parallel_size")

    @property
    def engine_rank_count(self) -> int:
        return self.tensor_parallel_size * self.data_parallel_size

    @property
    def unused_device_count(self) -> int:
        return self.physical_device_count - self.engine_rank_count

    def to_dict(self) -> dict[str, int]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "physical_device_count": self.physical_device_count,
            "independent_kv_ranks": self.independent_kv_ranks,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ParallelTopology:
        tp = _positive_int("tensor_parallel_size", data.get("tensor_parallel_size", 1))
        dp = _positive_int("data_parallel_size", data.get("data_parallel_size", 1))
        return cls(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            expert_parallel_size=_positive_int("expert_parallel_size", data.get("expert_parallel_size", 1)),
            physical_device_count=_positive_int("physical_device_count", data.get("physical_device_count", tp * dp)),
            independent_kv_ranks=_positive_int("independent_kv_ranks", data.get("independent_kv_ranks", dp)),
        )


@dataclass(frozen=True, slots=True)
class LLMDRoutingSpec:
    """llm-d limits that should agree with the serving runtime."""

    flow_control_token_limit: int | None = None
    max_concurrent_sequences: int | None = None
    kv_block_size_tokens: int | None = None
    output_ratio: float | None = None
    max_estimated_output_tokens: int | None = None
    precise_prefix_routing: bool = False

    def __post_init__(self) -> None:
        for name in (
            "flow_control_token_limit",
            "max_concurrent_sequences",
            "kv_block_size_tokens",
            "max_estimated_output_tokens",
        ):
            if getattr(self, name) is not None:
                _positive_int(name, getattr(self, name))
        if self.output_ratio is not None:
            ratio = _non_negative_float("output_ratio", self.output_ratio)
            if ratio > 1:
                raise ContractError("output_ratio cannot exceed 1")
        if not isinstance(self.precise_prefix_routing, bool):
            raise ContractError("precise_prefix_routing must be a boolean")
        if self.precise_prefix_routing and self.kv_block_size_tokens is None:
            raise ContractError("precise prefix routing requires kv_block_size_tokens")

    def to_dict(self) -> dict[str, Any]:
        return {
            "flow_control_token_limit": self.flow_control_token_limit,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "kv_block_size_tokens": self.kv_block_size_tokens,
            "output_ratio": self.output_ratio,
            "max_estimated_output_tokens": self.max_estimated_output_tokens,
            "precise_prefix_routing": self.precise_prefix_routing,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LLMDRoutingSpec:
        precise = data.get("precise_prefix_routing", False)
        if not isinstance(precise, bool):
            raise ContractError("precise_prefix_routing must be a boolean")
        return cls(
            flow_control_token_limit=_optional_positive_int(
                "flow_control_token_limit", data.get("flow_control_token_limit")
            ),
            max_concurrent_sequences=_optional_positive_int(
                "max_concurrent_sequences", data.get("max_concurrent_sequences")
            ),
            kv_block_size_tokens=_optional_positive_int("kv_block_size_tokens", data.get("kv_block_size_tokens")),
            output_ratio=(
                None
                if data.get("output_ratio") is None
                else _non_negative_float("output_ratio", data.get("output_ratio"))
            ),
            max_estimated_output_tokens=_optional_positive_int(
                "max_estimated_output_tokens", data.get("max_estimated_output_tokens")
            ),
            precise_prefix_routing=precise,
        )


@dataclass(frozen=True, slots=True)
class ServingRecipe:
    """Normalized model, host, runtime, topology, and routing configuration."""

    schema_version: str
    recipe_id: str
    model: ModelSpec
    host: HardwareSpec
    runtime: RuntimeVariant
    topology: ParallelTopology
    llmd: LLMDRoutingSpec
    configured_groups: int = 1
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "serving-recipe-1.0":
            raise ContractError("unsupported serving recipe schema_version")
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ContractError("recipe_id is required")
        _positive_int("configured_groups", self.configured_groups)
        if self.host.device_count != self.topology.physical_device_count:
            raise ContractError("host device_count must equal topology physical_device_count")
        if self.runtime.tensor_parallel_size != self.topology.tensor_parallel_size:
            raise ContractError("runtime and topology tensor_parallel_size must match")
        if self.runtime.max_num_seqs is None:
            raise ContractError("recipe runtime requires max_num_seqs")
        if self.topology.expert_parallel_size > 1 and self.runtime.weight_bytes_per_device_override is None:
            raise ContractError("expert-parallel recipes require weight_bytes_per_device_override")
        if not self.evidence:
            raise ContractError("serving recipes require provenance evidence")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "model": self.model.to_dict(),
            "host": self.host.to_dict(),
            "runtime": self.runtime.to_dict(),
            "topology": self.topology.to_dict(),
            "llmd": self.llmd.to_dict(),
            "configured_groups": self.configured_groups,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @property
    def variant_fingerprint(self) -> str:
        """Stable identity for the model, host, runtime, topology, and routing shape."""

        encoded = json.dumps(self._variant_document(), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _variant_document(self) -> dict[str, Any]:
        host = self.host.to_dict()
        host.pop("price_per_device_hour")
        host["memory_utilization_limit"] = float(host["memory_utilization_limit"])
        runtime = self.runtime.to_dict()
        runtime.pop("notes")
        runtime.pop("supported_vendors")
        llmd = self.llmd.to_dict()
        if llmd["output_ratio"] is not None:
            llmd["output_ratio"] = float(llmd["output_ratio"])
        return {
            "fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
            "model": self.model.to_dict(),
            "host": host,
            "runtime": runtime,
            "topology": self.topology.to_dict(),
            "llmd": llmd,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ServingRecipe:
        raw_evidence = _sequence("evidence", data.get("evidence", ()))
        return cls(
            schema_version=_required_str(data, "schema_version"),
            recipe_id=_required_str(data, "recipe_id"),
            model=ModelSpec.from_dict(_mapping("model", _required(data, "model"))),
            host=HardwareSpec.from_dict(_mapping("host", _required(data, "host"))),
            runtime=RuntimeVariant.from_dict(_mapping("runtime", _required(data, "runtime"))),
            topology=ParallelTopology.from_dict(_mapping("topology", _required(data, "topology"))),
            llmd=LLMDRoutingSpec.from_dict(_mapping("llmd", data.get("llmd", {}))),
            configured_groups=_positive_int("configured_groups", data.get("configured_groups", 1)),
            evidence=tuple(EvidenceRecord.from_dict(_mapping("evidence item", item)) for item in raw_evidence),
        )


@dataclass(frozen=True, slots=True)
class MeasuredGroupProfile:
    """Measured capacity and latency at one exact serving-group operating point."""

    profile_id: str
    recipe_variant_fingerprint_version: str
    recipe_variant_fingerprint: str
    context_tokens: int
    input_tokens_per_request: float = 0
    output_tokens_per_request: float = 0
    requests_per_second: float | None = None
    prefill_tokens_per_second: float | None = None
    decode_tokens_per_second: float | None = None
    concurrent_sequences: int | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    latency_percentile: float | None = None
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ContractError("profile_id is required")
        if self.recipe_variant_fingerprint_version != RECIPE_VARIANT_FINGERPRINT_VERSION:
            raise ContractError("unsupported recipe_variant_fingerprint_version")
        digest = (
            self.recipe_variant_fingerprint.removeprefix("sha256:")
            if isinstance(self.recipe_variant_fingerprint, str)
            else ""
        )
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ContractError("recipe_variant_fingerprint must be a sha256 fingerprint")
        _positive_int("context_tokens", self.context_tokens)
        for name in ("input_tokens_per_request", "output_tokens_per_request"):
            _non_negative_float(name, getattr(self, name))
        for name in (
            "requests_per_second",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "ttft_ms",
            "tpot_ms",
            "latency_percentile",
        ):
            if getattr(self, name) is not None:
                _optional_positive_float(name, getattr(self, name))
        if self.concurrent_sequences is not None:
            _positive_int("concurrent_sequences", self.concurrent_sequences)
        if (
            self.requests_per_second is not None
            and self.prefill_tokens_per_second is not None
            and self.input_tokens_per_request > 0
            and not isclose(
                self.prefill_tokens_per_second,
                self.requests_per_second * self.input_tokens_per_request,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ContractError("measured prefill TPS conflicts with requests per second x input tokens")
        if (
            self.requests_per_second is not None
            and self.decode_tokens_per_second is not None
            and self.output_tokens_per_request > 0
            and not isclose(
                self.decode_tokens_per_second,
                self.requests_per_second * self.output_tokens_per_request,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ContractError("measured decode TPS conflicts with requests per second x output tokens")
        measured_fields = {
            name: value
            for name, value in (
                ("requests_per_second", self.requests_per_second),
                ("prefill_tokens_per_second", self.prefill_tokens_per_second),
                ("decode_tokens_per_second", self.decode_tokens_per_second),
                ("concurrent_sequences", self.concurrent_sequences),
                ("ttft_ms", self.ttft_ms),
                ("tpot_ms", self.tpot_ms),
                ("latency_percentile", self.latency_percentile),
            )
            if value is not None
        }
        if not measured_fields:
            raise ContractError("a measured group profile requires at least one metric")
        measured_evidence = tuple(item for item in self.evidence if item.kind == EvidenceKind.MEASURED)
        if not measured_evidence:
            raise ContractError("a measured group profile requires measured evidence")
        operating_point = {
            "recipe_variant_fingerprint_version": self.recipe_variant_fingerprint_version,
            "recipe_variant_fingerprint": self.recipe_variant_fingerprint,
            "context_tokens": self.context_tokens,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
        }
        complete_claim = operating_point | measured_fields
        if not any(
            all(item.metrics.get(name) == value for name, value in complete_claim.items()) for item in measured_evidence
        ):
            raise ContractError(
                "one measured evidence record must bind the exact recipe, operating point, and every metric"
            )
        if (self.ttft_ms is not None or self.tpot_ms is not None) and not any(
            value is not None
            for value in (self.requests_per_second, self.prefill_tokens_per_second, self.decode_tokens_per_second)
        ):
            raise ContractError("latency measurements require a measured traffic operating point")
        if (self.ttft_ms is not None or self.tpot_ms is not None) and self.latency_percentile is None:
            raise ContractError("latency measurements require latency_percentile")
        if self.latency_percentile is not None and not 0 < self.latency_percentile < 100:
            raise ContractError("latency_percentile must be in (0, 100)")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "recipe_variant_fingerprint_version": self.recipe_variant_fingerprint_version,
            "recipe_variant_fingerprint": self.recipe_variant_fingerprint,
            "context_tokens": self.context_tokens,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
            "requests_per_second": self.requests_per_second,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "concurrent_sequences": self.concurrent_sequences,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "latency_percentile": self.latency_percentile,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeasuredGroupProfile:
        raw_evidence = _sequence("evidence", data.get("evidence", ()))
        return cls(
            profile_id=_required_str(data, "profile_id"),
            recipe_variant_fingerprint_version=_required_str(data, "recipe_variant_fingerprint_version"),
            recipe_variant_fingerprint=_required_str(data, "recipe_variant_fingerprint"),
            context_tokens=_positive_int("context_tokens", _required(data, "context_tokens")),
            input_tokens_per_request=_non_negative_float(
                "input_tokens_per_request", data.get("input_tokens_per_request", 0)
            ),
            output_tokens_per_request=_non_negative_float(
                "output_tokens_per_request", data.get("output_tokens_per_request", 0)
            ),
            requests_per_second=_optional_positive_float("requests_per_second", data.get("requests_per_second")),
            prefill_tokens_per_second=_optional_positive_float(
                "prefill_tokens_per_second", data.get("prefill_tokens_per_second")
            ),
            decode_tokens_per_second=_optional_positive_float(
                "decode_tokens_per_second", data.get("decode_tokens_per_second")
            ),
            concurrent_sequences=_optional_positive_int("concurrent_sequences", data.get("concurrent_sequences")),
            ttft_ms=_optional_positive_float("ttft_ms", data.get("ttft_ms")),
            tpot_ms=_optional_positive_float("tpot_ms", data.get("tpot_ms")),
            latency_percentile=_optional_positive_float("latency_percentile", data.get("latency_percentile")),
            evidence=tuple(EvidenceRecord.from_dict(_mapping("evidence item", item)) for item in raw_evidence),
        )


@dataclass(frozen=True, slots=True)
class LoadRequirement:
    """Requested load plus an optional exact-variant measured group profile."""

    schema_version: str
    context_tokens: int
    peak_concurrent_sequences: int = 0
    request_rate_per_second: float = 0
    input_tokens_per_request: float = 0
    output_tokens_per_request: float = 0
    prefill_tokens_per_second: float = 0
    decode_tokens_per_second: float = 0
    target_utilization: float = 0.8
    target_ttft_ms: float | None = None
    target_tpot_ms: float | None = None
    target_latency_percentile: float | None = None
    measured_profile: MeasuredGroupProfile | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "load-requirement-1.0":
            raise ContractError("unsupported load requirement schema_version")
        _positive_int("context_tokens", self.context_tokens)
        if (
            not isinstance(self.peak_concurrent_sequences, int)
            or isinstance(self.peak_concurrent_sequences, bool)
            or self.peak_concurrent_sequences < 0
        ):
            raise ContractError("peak_concurrent_sequences must be a non-negative integer")
        for name in (
            "request_rate_per_second",
            "input_tokens_per_request",
            "output_tokens_per_request",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
        ):
            _non_negative_float(name, getattr(self, name))
        for name in ("target_ttft_ms", "target_tpot_ms", "target_latency_percentile"):
            if getattr(self, name) is not None:
                _optional_positive_float(name, getattr(self, name))
        if (
            self.target_ttft_ms is not None or self.target_tpot_ms is not None
        ) and self.target_latency_percentile is None:
            raise ContractError("latency targets require target_latency_percentile")
        if self.target_latency_percentile is not None and not 0 < self.target_latency_percentile < 100:
            raise ContractError("target_latency_percentile must be in (0, 100)")
        utilization = _optional_positive_float("target_utilization", self.target_utilization)
        if utilization is None or utilization > 1:
            raise ContractError("target_utilization must be in (0, 1]")
        derived_prefill = self.request_rate_per_second * self.input_tokens_per_request
        derived_decode = self.request_rate_per_second * self.output_tokens_per_request
        if (
            self.prefill_tokens_per_second
            and derived_prefill
            and not isclose(self.prefill_tokens_per_second, derived_prefill, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ContractError("prefill TPS conflicts with request rate x input tokens per request")
        if (
            self.decode_tokens_per_second
            and derived_decode
            and not isclose(self.decode_tokens_per_second, derived_decode, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ContractError("decode TPS conflicts with request rate x output tokens per request")

    @property
    def effective_prefill_tokens_per_second(self) -> float:
        return self.prefill_tokens_per_second or self.request_rate_per_second * self.input_tokens_per_request

    @property
    def effective_decode_tokens_per_second(self) -> float:
        return self.decode_tokens_per_second or self.request_rate_per_second * self.output_tokens_per_request

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_tokens": self.context_tokens,
            "peak_concurrent_sequences": self.peak_concurrent_sequences,
            "request_rate_per_second": self.request_rate_per_second,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "target_utilization": self.target_utilization,
            "target_ttft_ms": self.target_ttft_ms,
            "target_tpot_ms": self.target_tpot_ms,
            "target_latency_percentile": self.target_latency_percentile,
            "measured_profile": None if self.measured_profile is None else self.measured_profile.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LoadRequirement:
        raw_profile = data.get("measured_profile")
        return cls(
            schema_version=_required_str(data, "schema_version"),
            context_tokens=_positive_int("context_tokens", _required(data, "context_tokens")),
            peak_concurrent_sequences=data.get("peak_concurrent_sequences", 0),
            request_rate_per_second=_non_negative_float(
                "request_rate_per_second", data.get("request_rate_per_second", 0)
            ),
            input_tokens_per_request=_non_negative_float(
                "input_tokens_per_request", data.get("input_tokens_per_request", 0)
            ),
            output_tokens_per_request=_non_negative_float(
                "output_tokens_per_request", data.get("output_tokens_per_request", 0)
            ),
            prefill_tokens_per_second=_non_negative_float(
                "prefill_tokens_per_second", data.get("prefill_tokens_per_second", 0)
            ),
            decode_tokens_per_second=_non_negative_float(
                "decode_tokens_per_second", data.get("decode_tokens_per_second", 0)
            ),
            target_utilization=_optional_positive_float("target_utilization", data.get("target_utilization", 0.8))
            or 0.8,
            target_ttft_ms=_optional_positive_float("target_ttft_ms", data.get("target_ttft_ms")),
            target_tpot_ms=_optional_positive_float("target_tpot_ms", data.get("target_tpot_ms")),
            target_latency_percentile=_optional_positive_float(
                "target_latency_percentile", data.get("target_latency_percentile")
            ),
            measured_profile=(
                None
                if raw_profile is None
                else MeasuredGroupProfile.from_dict(_mapping("measured_profile", raw_profile))
            ),
        )


class AuditStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class RecipeAudit:
    schema_version: str
    formula_version: str
    recipe_variant_fingerprint_version: str
    recipe_variant_fingerprint: str
    recipe: ServingRecipe
    load: LoadRequirement
    rank_contract: CapacityContract
    status: AuditStatus
    context_supported: bool
    analytical_sequences_per_kv_rank: int
    analytical_sequences_per_group: int
    calculated_kv_tokens_per_group: int
    calculated_max_concurrent_sequences: int
    configured_devices: int
    required_groups: int | None
    required_devices: int | None
    additional_groups_needed: int | None
    additional_devices_needed: int | None
    drivers: Mapping[str, int]
    configured_load_satisfied: bool | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    suggestions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "formula_version": self.formula_version,
            "recipe_variant_fingerprint_version": self.recipe_variant_fingerprint_version,
            "recipe_variant_fingerprint": self.recipe_variant_fingerprint,
            "recipe": self.recipe.to_dict(),
            "load": self.load.to_dict(),
            "rank_contract": self.rank_contract.to_dict(),
            "status": self.status.value,
            "context_supported": self.context_supported,
            "analytical_sequences_per_kv_rank": self.analytical_sequences_per_kv_rank,
            "analytical_sequences_per_group": self.analytical_sequences_per_group,
            "calculated_kv_tokens_per_group": self.calculated_kv_tokens_per_group,
            "calculated_max_concurrent_sequences": self.calculated_max_concurrent_sequences,
            "configured_devices": self.configured_devices,
            "required_groups": self.required_groups,
            "required_devices": self.required_devices,
            "additional_groups_needed": self.additional_groups_needed,
            "additional_devices_needed": self.additional_devices_needed,
            "drivers": dict(self.drivers),
            "configured_load_satisfied": self.configured_load_satisfied,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
        }


def _groups_for(demand: float, capacity: float, utilization: float) -> int:
    ratio = Decimal(str(demand)) / (Decimal(str(capacity)) * Decimal(str(utilization)))
    return max(1, int(ratio.to_integral_value(rounding=ROUND_CEILING)))


def audit_recipe(recipe: ServingRecipe, load: LoadRequirement) -> RecipeAudit:
    """Check one normalized serving recipe against memory, routing, and load."""

    rank_hardware = replace(
        recipe.host,
        device_count=recipe.topology.tensor_parallel_size,
    )
    contract = capacity_for(recipe.model, rank_hardware, recipe.runtime)
    topology = recipe.topology
    llmd = recipe.llmd
    issues: list[str] = []
    configuration_issues: list[str] = []
    warnings = list(contract.warnings)
    suggestions: list[str] = []

    if not contract.fits:
        configuration_issues.append("the model and declared reserves do not fit on one KV rank")
    context_supported = contract.fits and load.context_tokens <= contract.max_context_tokens
    if not context_supported:
        issues.append("the requested context exceeds the calculated context capacity")

    per_rank_sequences = contract.max_sequences_at(load.context_tokens) if context_supported else 0
    per_group_sequences = per_rank_sequences * topology.independent_kv_ranks
    kv_tokens_per_group = contract.kv_capacity_tokens_per_device * topology.independent_kv_ranks
    configured_max_sequences = (recipe.runtime.max_num_seqs or 0) * topology.independent_kv_ranks

    if topology.unused_device_count:
        warnings.append(
            f"the host has {topology.unused_device_count} physical devices not assigned to TP x DP engine ranks"
        )
    if llmd.kv_block_size_tokens is not None and llmd.kv_block_size_tokens != recipe.runtime.block_size_tokens:
        configuration_issues.append("llm-d and vLLM block sizes do not match")
    if llmd.precise_prefix_routing and llmd.kv_block_size_tokens is None:
        configuration_issues.append("precise prefix routing is missing its KV block size")
    if llmd.flow_control_token_limit is not None and llmd.flow_control_token_limit > kv_tokens_per_group:
        configuration_issues.append("llm-d flow-control token limit exceeds calculated KV capacity")
    if llmd.max_concurrent_sequences is not None and llmd.max_concurrent_sequences != configured_max_sequences:
        configuration_issues.append("llm-d max concurrency does not match max_num_seqs x independent KV ranks")

    profile = load.measured_profile
    profile_issues: list[str] = []
    if profile is not None:
        if profile.recipe_variant_fingerprint != recipe.variant_fingerprint:
            profile_issues.append("the measured profile does not match the serving recipe fingerprint")
        if profile.context_tokens != load.context_tokens:
            profile_issues.append("the measured profile context does not match the requested context")
        if load.input_tokens_per_request and profile.input_tokens_per_request != load.input_tokens_per_request:
            profile_issues.append("the measured profile input shape does not match the requested load")
        if load.output_tokens_per_request and profile.output_tokens_per_request != load.output_tokens_per_request:
            profile_issues.append("the measured profile output shape does not match the requested load")
        if (
            load.target_latency_percentile is not None
            and (profile.ttft_ms is not None or profile.tpot_ms is not None)
            and profile.latency_percentile != load.target_latency_percentile
        ):
            profile_issues.append("the measured latency percentile does not match the requested percentile")
    configuration_issues.extend(profile_issues)
    usable_profile = profile if profile is not None and not profile_issues else None

    drivers: dict[str, int] = {}
    incomplete: list[str] = []
    if load.peak_concurrent_sequences > 0:
        analytical_limit = per_group_sequences
        if usable_profile is not None and usable_profile.concurrent_sequences is not None:
            analytical_limit = min(analytical_limit, usable_profile.concurrent_sequences)
        if analytical_limit <= 0:
            issues.append("the recipe has no sequence capacity at the requested context")
        else:
            drivers["peak_concurrent_sequences"] = _groups_for(
                float(load.peak_concurrent_sequences),
                float(analytical_limit),
                load.target_utilization,
            )

    measured_drivers = (
        (
            "requests_per_second",
            load.request_rate_per_second,
            None if usable_profile is None else usable_profile.requests_per_second,
        ),
        (
            "prefill_tokens_per_second",
            load.effective_prefill_tokens_per_second,
            None if usable_profile is None else usable_profile.prefill_tokens_per_second,
        ),
        (
            "decode_tokens_per_second",
            load.effective_decode_tokens_per_second,
            None if usable_profile is None else usable_profile.decode_tokens_per_second,
        ),
    )
    for name, demand, capacity in measured_drivers:
        if demand <= 0:
            continue
        if capacity is None:
            incomplete.append(name)
        else:
            drivers[name] = _groups_for(demand, capacity, load.target_utilization)

    for name, target, observed in (
        ("TTFT", load.target_ttft_ms, None if usable_profile is None else usable_profile.ttft_ms),
        ("TPOT", load.target_tpot_ms, None if usable_profile is None else usable_profile.tpot_ms),
    ):
        if target is None:
            continue
        if observed is None:
            incomplete.append(name)
        elif observed > target:
            issues.append(f"observed {name} exceeds the requested target")

    required_groups = max(drivers.values(), default=1) if not incomplete else None
    required_devices = None if required_groups is None else required_groups * topology.physical_device_count
    configured_devices = recipe.configured_groups * topology.physical_device_count
    additional_groups_needed = None if required_groups is None else max(0, required_groups - recipe.configured_groups)
    additional_devices_needed = (
        None if additional_groups_needed is None else additional_groups_needed * topology.physical_device_count
    )
    configured_load_satisfied: bool | None
    issues.extend(configuration_issues)
    if issues:
        configured_load_satisfied = False
    elif incomplete:
        configured_load_satisfied = None
    elif required_groups is not None:
        configured_load_satisfied = recipe.configured_groups >= required_groups
        if not configured_load_satisfied:
            issues.append(f"configured_groups={recipe.configured_groups} is below required_groups={required_groups}")
    else:
        configured_load_satisfied = None

    if incomplete:
        warnings.extend(f"{name} demand or SLO has no matching measured capacity" for name in incomplete)
        suggestions.append("attach a measured profile for every non-zero traffic or latency requirement")
    if llmd.flow_control_token_limit is None:
        suggestions.append("set and audit an llm-d flow-control token limit")
    elif llmd.flow_control_token_limit < kv_tokens_per_group:
        suggestions.append("validate the llm-d flow-control headroom against queueing and latency measurements")
    if topology.expert_parallel_size > 1:
        suggestions.append("refresh per-device resident weight bytes whenever the EP layout or runtime image changes")

    if configuration_issues:
        status = AuditStatus.INVALID
    elif configured_load_satisfied is False:
        status = AuditStatus.INSUFFICIENT
    elif configured_load_satisfied is None:
        status = AuditStatus.INCOMPLETE
    else:
        status = AuditStatus.SUFFICIENT

    return RecipeAudit(
        schema_version="recipe-audit-1.0",
        formula_version="recipe-audit-formula-1.0",
        recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
        recipe_variant_fingerprint=recipe.variant_fingerprint,
        recipe=recipe,
        load=load,
        rank_contract=contract,
        status=status,
        context_supported=context_supported,
        analytical_sequences_per_kv_rank=per_rank_sequences,
        analytical_sequences_per_group=per_group_sequences,
        calculated_kv_tokens_per_group=kv_tokens_per_group,
        calculated_max_concurrent_sequences=configured_max_sequences,
        configured_devices=configured_devices,
        required_groups=required_groups,
        required_devices=required_devices,
        additional_groups_needed=additional_groups_needed,
        additional_devices_needed=additional_devices_needed,
        drivers=drivers,
        configured_load_satisfied=configured_load_satisfied,
        issues=tuple(issues),
        warnings=tuple(warnings),
        suggestions=tuple(suggestions),
    )
