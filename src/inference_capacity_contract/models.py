"""Public domain objects for deterministic inference-capacity contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import ceil
from typing import Any


class ContractError(ValueError):
    """Raised when an input cannot produce a meaningful contract."""


class ValidationLevel(StrEnum):
    ANALYTICALLY_FEASIBLE = "analytically-feasible"
    RUNTIME_SUPPORTED = "runtime-supported"
    INITIALIZATION_VALIDATED = "initialization-validated"
    SLO_VALIDATED = "slo-validated"


class EvidenceKind(StrEnum):
    ANALYTICAL = "analytical"
    MEASURED = "measured"
    REPORTED = "reported"
    EXTRAPOLATED = "extrapolated"


WEIGHT_BYTES: dict[str, float] = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


def _positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ContractError(f"{name} must be positive, got {value!r}")


def _fraction(name: str, value: float) -> None:
    if not 0 < value <= 1:
        raise ContractError(f"{name} must be in (0, 1], got {value!r}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _required_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Architecture and resident-weight facts for one immutable artifact."""

    model_id: str
    revision: str
    parameter_count: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    weight_dtype: str = "bf16"
    max_model_len: int | None = None
    explicit_weight_bytes: int | None = None
    attention_type: str = "gqa"
    kv_bytes_per_token_per_device_override: int | None = None
    architecture: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ContractError("model_id and revision are required")
        for name in ("parameter_count", "num_layers", "num_kv_heads", "head_dim"):
            _positive(name, int(getattr(self, name)))
        if self.weight_dtype not in WEIGHT_BYTES and self.explicit_weight_bytes is None:
            raise ContractError(f"unsupported weight_dtype {self.weight_dtype!r}; provide explicit_weight_bytes")
        if self.max_model_len is not None:
            _positive("max_model_len", self.max_model_len)
        if self.explicit_weight_bytes is not None:
            _positive("explicit_weight_bytes", self.explicit_weight_bytes)
        if self.kv_bytes_per_token_per_device_override is not None:
            _positive(
                "kv_bytes_per_token_per_device_override",
                self.kv_bytes_per_token_per_device_override,
            )
        if self.attention_type not in {"mha", "gqa", "mqa", "mla", "custom"}:
            raise ContractError(f"unsupported attention_type {self.attention_type!r}")

    @property
    def weight_bytes(self) -> int:
        if self.explicit_weight_bytes is not None:
            return self.explicit_weight_bytes
        return int(self.parameter_count * WEIGHT_BYTES[self.weight_dtype])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "parameter_count": self.parameter_count,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "weight_dtype": self.weight_dtype,
            "max_model_len": self.max_model_len,
            "explicit_weight_bytes": self.explicit_weight_bytes,
            "attention_type": self.attention_type,
            "kv_bytes_per_token_per_device_override": (self.kv_bytes_per_token_per_device_override),
            "architecture": self.architecture,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelSpec:
        override = data.get("kv_bytes_per_token_per_device_override")
        if override is None:
            override = data.get("kv_bytes_per_token_override")
        return cls(
            model_id=str(data["model_id"]),
            revision=str(data["revision"]),
            parameter_count=int(data["parameter_count"]),
            num_layers=int(data["num_layers"]),
            num_kv_heads=int(data["num_kv_heads"]),
            head_dim=int(data["head_dim"]),
            weight_dtype=str(data.get("weight_dtype", "bf16")),
            max_model_len=_optional_int(data.get("max_model_len")),
            explicit_weight_bytes=_optional_int(data.get("explicit_weight_bytes")),
            attention_type=str(data.get("attention_type", "gqa")),
            kv_bytes_per_token_per_device_override=_optional_int(override),
            architecture=_optional_str(data.get("architecture")),
        )


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """Hardware topology allocated to one tensor-parallel serving replica."""

    hardware_id: str
    vendor: str
    device_count: int
    memory_bytes_per_device: int
    memory_utilization_limit: float = 0.90
    price_per_device_hour: float | None = None
    interconnect: str | None = None

    def __post_init__(self) -> None:
        if not self.hardware_id or not self.vendor:
            raise ContractError("hardware_id and vendor are required")
        _positive("device_count", self.device_count)
        _positive("memory_bytes_per_device", self.memory_bytes_per_device)
        _fraction("memory_utilization_limit", self.memory_utilization_limit)
        if self.price_per_device_hour is not None and self.price_per_device_hour < 0:
            raise ContractError("price_per_device_hour cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "vendor": self.vendor,
            "device_count": self.device_count,
            "memory_bytes_per_device": self.memory_bytes_per_device,
            "memory_utilization_limit": self.memory_utilization_limit,
            "price_per_device_hour": self.price_per_device_hour,
            "interconnect": self.interconnect,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareSpec:
        return cls(
            hardware_id=str(data["hardware_id"]),
            vendor=str(data["vendor"]),
            device_count=int(data["device_count"]),
            memory_bytes_per_device=int(data["memory_bytes_per_device"]),
            memory_utilization_limit=float(data.get("memory_utilization_limit", 0.90)),
            price_per_device_hour=_optional_float(data.get("price_per_device_hour")),
            interconnect=_optional_str(data.get("interconnect")),
        )


@dataclass(frozen=True, slots=True)
class RuntimeVariant:
    """Pinned runtime configuration for one tensor-parallel serving replica.

    Replica/data parallel count is intentionally not part of this object. A
    planner creates multiple replicas from a per-replica capacity contract.
    For tensor parallelism greater than one, the runtime adapter must declare
    the number of KV heads resident on each device.
    """

    engine: str
    version: str
    tensor_parallel_size: int = 1
    kv_cache_dtype: str = "bf16"
    kv_heads_per_device: int | None = None
    runtime_overhead_bytes_per_device: int = 0
    activation_reserve_bytes_per_device: int = 0
    max_num_seqs: int | None = None
    block_size_tokens: int = 16
    supported_vendors: tuple[str, ...] = ("nvidia", "amd")
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.engine not in {"vllm", "sglang", "other"}:
            raise ContractError(f"unsupported engine {self.engine!r}")
        if not self.version:
            raise ContractError("runtime version is required")
        for name in ("tensor_parallel_size", "block_size_tokens"):
            _positive(name, int(getattr(self, name)))
        if self.kv_heads_per_device is not None:
            _positive("kv_heads_per_device", self.kv_heads_per_device)
        for name in (
            "runtime_overhead_bytes_per_device",
            "activation_reserve_bytes_per_device",
        ):
            if int(getattr(self, name)) < 0:
                raise ContractError(f"{name} cannot be negative")
        if self.max_num_seqs is not None:
            _positive("max_num_seqs", self.max_num_seqs)
        if self.kv_cache_dtype not in WEIGHT_BYTES:
            raise ContractError(f"unsupported kv_cache_dtype {self.kv_cache_dtype!r}")
        if not self.supported_vendors:
            raise ContractError("supported_vendors cannot be empty")

    @property
    def devices_required(self) -> int:
        return self.tensor_parallel_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "version": self.version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "kv_cache_dtype": self.kv_cache_dtype,
            "kv_heads_per_device": self.kv_heads_per_device,
            "runtime_overhead_bytes_per_device": self.runtime_overhead_bytes_per_device,
            "activation_reserve_bytes_per_device": self.activation_reserve_bytes_per_device,
            "max_num_seqs": self.max_num_seqs,
            "block_size_tokens": self.block_size_tokens,
            "supported_vendors": list(self.supported_vendors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeVariant:
        legacy_dp = int(data.get("data_parallel_size", 1))
        if legacy_dp != 1:
            raise ContractError(
                "data_parallel_size is not part of a per-replica runtime variant; "
                "create multiple capacity contracts or replicas instead"
            )
        return cls(
            engine=str(data["engine"]),
            version=str(data["version"]),
            tensor_parallel_size=int(data.get("tensor_parallel_size", 1)),
            kv_cache_dtype=str(data.get("kv_cache_dtype", "bf16")),
            kv_heads_per_device=_optional_int(data.get("kv_heads_per_device")),
            runtime_overhead_bytes_per_device=int(data.get("runtime_overhead_bytes_per_device", 0)),
            activation_reserve_bytes_per_device=int(data.get("activation_reserve_bytes_per_device", 0)),
            max_num_seqs=_optional_int(data.get("max_num_seqs")),
            block_size_tokens=int(data.get("block_size_tokens", 16)),
            supported_vendors=tuple(str(item) for item in data.get("supported_vendors", ("nvidia", "amd"))),
            notes=tuple(str(item) for item in data.get("notes", ())),
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Provenance attached to a capacity claim."""

    evidence_id: str
    kind: EvidenceKind
    source: str
    scope: str
    metrics: Mapping[str, float | int | str] = field(default_factory=dict)
    collected_at: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source or not self.scope:
            raise ContractError("evidence_id, source, and scope are required")
        try:
            normalized_kind = EvidenceKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"unsupported evidence kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", normalized_kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "source": self.source,
            "scope": self.scope,
            "metrics": dict(self.metrics),
            "collected_at": self.collected_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        metrics = data.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ContractError("evidence metrics must be an object")
        return cls(
            evidence_id=str(data["evidence_id"]),
            kind=EvidenceKind(str(data["kind"])),
            source=str(data["source"]),
            scope=str(data["scope"]),
            metrics={str(key): value for key, value in metrics.items()},
            collected_at=_optional_str(data.get("collected_at")),
        )


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    """Measured demand and per-replica capacity for one exact variant."""

    profile_id: str
    model_id: str
    model_revision: str
    hardware_id: str
    runtime_engine: str
    runtime_version: str
    request_rate_per_second: float
    input_tokens_per_request: float = 0.0
    output_tokens_per_request: float = 0.0
    sustainable_requests_per_replica_per_second: float | None = None
    sustainable_prefill_tokens_per_replica_per_second: float | None = None
    sustainable_decode_tokens_per_replica_per_second: float | None = None
    sustainable_concurrent_sequences_per_replica: int | None = None
    peak_concurrent_sequences: int | None = None
    concurrency_context_tokens: int | None = None
    target_utilization: float = 0.70
    scale_up_buffer: float = 1.15
    min_replicas: int = 1
    max_replicas: int | None = None
    baseline_replicas: int | None = None
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        identities = (
            self.profile_id,
            self.model_id,
            self.model_revision,
            self.hardware_id,
            self.runtime_engine,
            self.runtime_version,
        )
        if not all(identities):
            raise ContractError("profile and variant identity fields are required")
        for name in (
            "request_rate_per_second",
            "input_tokens_per_request",
            "output_tokens_per_request",
        ):
            if float(getattr(self, name)) < 0:
                raise ContractError(f"{name} cannot be negative")
        sustainable_fields = (
            "sustainable_requests_per_replica_per_second",
            "sustainable_prefill_tokens_per_replica_per_second",
            "sustainable_decode_tokens_per_replica_per_second",
            "sustainable_concurrent_sequences_per_replica",
        )
        if all(getattr(self, name) is None for name in sustainable_fields):
            raise ContractError("at least one sustainable per-replica capacity is required")
        for name in sustainable_fields:
            value = getattr(self, name)
            if value is not None and float(value) <= 0:
                raise ContractError(f"{name} must be positive when provided")
        _fraction("target_utilization", self.target_utilization)
        if self.scale_up_buffer < 1:
            raise ContractError("scale_up_buffer must be at least 1")
        _positive("min_replicas", self.min_replicas)
        if self.max_replicas is not None:
            _positive("max_replicas", self.max_replicas)
            if self.max_replicas < self.min_replicas:
                raise ContractError("max_replicas cannot be lower than min_replicas")
        if self.baseline_replicas is not None:
            _positive("baseline_replicas", self.baseline_replicas)
        if self.peak_concurrent_sequences is not None:
            _positive("peak_concurrent_sequences", self.peak_concurrent_sequences)
            if self.concurrency_context_tokens is None:
                raise ContractError("concurrency_context_tokens is required with peak_concurrent_sequences")
            if self.sustainable_concurrent_sequences_per_replica is None:
                raise ContractError(
                    "sustainable_concurrent_sequences_per_replica is required with peak_concurrent_sequences"
                )
        if self.concurrency_context_tokens is not None:
            _positive("concurrency_context_tokens", self.concurrency_context_tokens)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "hardware_id": self.hardware_id,
            "runtime_engine": self.runtime_engine,
            "runtime_version": self.runtime_version,
            "request_rate_per_second": self.request_rate_per_second,
            "input_tokens_per_request": self.input_tokens_per_request,
            "output_tokens_per_request": self.output_tokens_per_request,
            "sustainable_requests_per_replica_per_second": (self.sustainable_requests_per_replica_per_second),
            "sustainable_prefill_tokens_per_replica_per_second": (
                self.sustainable_prefill_tokens_per_replica_per_second
            ),
            "sustainable_decode_tokens_per_replica_per_second": (self.sustainable_decode_tokens_per_replica_per_second),
            "sustainable_concurrent_sequences_per_replica": self.sustainable_concurrent_sequences_per_replica,
            "peak_concurrent_sequences": self.peak_concurrent_sequences,
            "concurrency_context_tokens": self.concurrency_context_tokens,
            "target_utilization": self.target_utilization,
            "scale_up_buffer": self.scale_up_buffer,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "baseline_replicas": self.baseline_replicas,
            "evidence": [record.to_dict() for record in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkloadProfile:
        raw_evidence = data.get("evidence", ())
        return cls(
            profile_id=str(data["profile_id"]),
            model_id=str(data["model_id"]),
            model_revision=str(data["model_revision"]),
            hardware_id=str(data["hardware_id"]),
            runtime_engine=str(data["runtime_engine"]),
            runtime_version=str(data["runtime_version"]),
            request_rate_per_second=float(data["request_rate_per_second"]),
            input_tokens_per_request=float(data.get("input_tokens_per_request", 0.0)),
            output_tokens_per_request=float(data.get("output_tokens_per_request", 0.0)),
            sustainable_requests_per_replica_per_second=_optional_float(
                data.get("sustainable_requests_per_replica_per_second")
            ),
            sustainable_prefill_tokens_per_replica_per_second=_optional_float(
                data.get("sustainable_prefill_tokens_per_replica_per_second")
            ),
            sustainable_decode_tokens_per_replica_per_second=_optional_float(
                data.get("sustainable_decode_tokens_per_replica_per_second")
            ),
            sustainable_concurrent_sequences_per_replica=_optional_int(
                data.get("sustainable_concurrent_sequences_per_replica")
            ),
            peak_concurrent_sequences=_optional_int(data.get("peak_concurrent_sequences")),
            concurrency_context_tokens=_optional_int(data.get("concurrency_context_tokens")),
            target_utilization=float(data.get("target_utilization", 0.70)),
            scale_up_buffer=float(data.get("scale_up_buffer", 1.15)),
            min_replicas=int(data.get("min_replicas", 1)),
            max_replicas=_optional_int(data.get("max_replicas")),
            baseline_replicas=_optional_int(data.get("baseline_replicas")),
            evidence=tuple(EvidenceRecord.from_dict(item) for item in raw_evidence),
        )


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    """Named hardware options available for reverse-fit queries."""

    items: tuple[HardwareSpec, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ContractError("hardware inventory cannot be empty")
        ids = [item.hardware_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ContractError("hardware inventory contains duplicate hardware_id values")

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HardwareInventory:
        return cls(tuple(HardwareSpec.from_dict(item) for item in data["items"]))


@dataclass(frozen=True, slots=True)
class ConcurrencyPoint:
    """Maximum sequence count at one active-token length."""

    context_tokens: int
    blocks_per_sequence: int
    max_sequences: int

    def __post_init__(self) -> None:
        _positive("context_tokens", self.context_tokens)
        _positive("blocks_per_sequence", self.blocks_per_sequence)
        if self.max_sequences < 0:
            raise ContractError("max_sequences cannot be negative")

    def to_dict(self) -> dict[str, int]:
        return {
            "context_tokens": self.context_tokens,
            "blocks_per_sequence": self.blocks_per_sequence,
            "max_sequences": self.max_sequences,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConcurrencyPoint:
        return cls(
            context_tokens=int(data["context_tokens"]),
            blocks_per_sequence=int(data["blocks_per_sequence"]),
            max_sequences=int(data["max_sequences"]),
        )


@dataclass(frozen=True, slots=True)
class CapacityContract:
    """A descriptive, versioned capacity result; never an actuation command."""

    schema_version: str
    model: ModelSpec
    hardware: HardwareSpec
    runtime: RuntimeVariant
    fits: bool
    validation_level: ValidationLevel
    memory_bytes_per_device: Mapping[str, int]
    kv_bytes_per_token_per_device: int
    kv_block_size_tokens: int
    kv_capacity_blocks_per_device: int
    kv_capacity_tokens_per_device: int
    max_context_tokens: int
    concurrency_envelope: tuple[ConcurrencyPoint, ...]
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "capacity-contract-2.0":
            raise ContractError(f"unsupported schema_version {self.schema_version!r}; expected 'capacity-contract-2.0'")
        try:
            normalized_level = ValidationLevel(self.validation_level)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"unsupported validation_level {self.validation_level!r}") from exc
        object.__setattr__(self, "validation_level", normalized_level)

        expected_memory_keys = {
            "usable_budget",
            "weights",
            "runtime_reserve",
            "activation_reserve",
            "kv_available",
        }
        if set(self.memory_bytes_per_device) != expected_memory_keys:
            raise ContractError("memory_bytes_per_device must contain the complete per-device memory ledger")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.memory_bytes_per_device.values()
        ):
            raise ContractError("memory_bytes_per_device values must be non-negative integers")
        _positive("kv_bytes_per_token_per_device", self.kv_bytes_per_token_per_device)
        _positive("kv_block_size_tokens", self.kv_block_size_tokens)
        if self.kv_capacity_blocks_per_device < 0 or self.kv_capacity_tokens_per_device < 0:
            raise ContractError("KV capacity cannot be negative")
        if self.max_context_tokens < 0:
            raise ContractError("max_context_tokens cannot be negative")
        if self.kv_capacity_tokens_per_device != (self.kv_capacity_blocks_per_device * self.kv_block_size_tokens):
            raise ContractError("KV token capacity must equal block capacity times block size")
        if self.max_context_tokens > self.kv_capacity_tokens_per_device:
            raise ContractError("max_context_tokens cannot exceed per-device KV token capacity")
        if self.fits and (self.kv_capacity_blocks_per_device == 0 or self.max_context_tokens == 0):
            raise ContractError("a fitting contract must have positive KV and context capacity")
        if not self.fits and (self.max_context_tokens != 0 or self.concurrency_envelope):
            raise ContractError("a non-fitting contract must have zero context capacity and no concurrency envelope")

        previous_context = 0
        for point in self.concurrency_envelope:
            if point.context_tokens <= previous_context:
                raise ContractError("concurrency_envelope context points must be strictly increasing")
            if point.context_tokens > self.max_context_tokens:
                raise ContractError("concurrency_envelope cannot exceed max_context_tokens")
            expected_blocks = ceil(point.context_tokens / self.kv_block_size_tokens)
            if point.blocks_per_sequence != expected_blocks:
                raise ContractError("concurrency_envelope blocks_per_sequence is inconsistent with block size")
            expected_sequences = self.kv_capacity_blocks_per_device // expected_blocks
            if self.runtime.max_num_seqs is not None:
                expected_sequences = min(expected_sequences, self.runtime.max_num_seqs)
            if point.max_sequences != expected_sequences:
                raise ContractError("concurrency_envelope max_sequences is inconsistent with KV capacity")
            previous_context = point.context_tokens

        object.__setattr__(self, "concurrency_envelope", tuple(self.concurrency_envelope))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def max_sequences_at(self, context_tokens: int) -> int:
        """Return the KV-memory sequence bound for an active-token length."""

        _positive("context_tokens", context_tokens)
        if not self.fits or context_tokens > self.max_context_tokens:
            return 0
        blocks_per_sequence = ceil(context_tokens / self.kv_block_size_tokens)
        sequences = self.kv_capacity_blocks_per_device // blocks_per_sequence
        if self.runtime.max_num_seqs is not None:
            sequences = min(sequences, self.runtime.max_num_seqs)
        return max(0, sequences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "hardware": self.hardware.to_dict(),
            "runtime": self.runtime.to_dict(),
            "fits": self.fits,
            "validation_level": self.validation_level.value,
            "memory_bytes_per_device": dict(self.memory_bytes_per_device),
            "kv_bytes_per_token_per_device": self.kv_bytes_per_token_per_device,
            "kv_block_size_tokens": self.kv_block_size_tokens,
            "kv_capacity_blocks_per_device": self.kv_capacity_blocks_per_device,
            "kv_capacity_tokens_per_device": self.kv_capacity_tokens_per_device,
            "max_context_tokens": self.max_context_tokens,
            "concurrency_envelope": [point.to_dict() for point in self.concurrency_envelope],
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def with_evidence(self, *records: EvidenceRecord) -> CapacityContract:
        """Return a copy with provenance attached without promoting validity."""

        if not records:
            return self
        return replace(self, evidence=self.evidence + tuple(records))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CapacityContract:
        schema_version = str(data["schema_version"])
        if schema_version != "capacity-contract-2.0":
            raise ContractError(f"unsupported schema_version {schema_version!r}; expected 'capacity-contract-2.0'")
        return cls(
            schema_version=schema_version,
            model=ModelSpec.from_dict(data["model"]),
            hardware=HardwareSpec.from_dict(data["hardware"]),
            runtime=RuntimeVariant.from_dict(data["runtime"]),
            fits=_required_bool("fits", data["fits"]),
            validation_level=ValidationLevel(str(data["validation_level"])),
            memory_bytes_per_device={str(key): int(value) for key, value in data["memory_bytes_per_device"].items()},
            kv_bytes_per_token_per_device=int(data["kv_bytes_per_token_per_device"]),
            kv_block_size_tokens=int(data["kv_block_size_tokens"]),
            kv_capacity_blocks_per_device=int(data["kv_capacity_blocks_per_device"]),
            kv_capacity_tokens_per_device=int(data["kv_capacity_tokens_per_device"]),
            max_context_tokens=int(data["max_context_tokens"]),
            concurrency_envelope=tuple(ConcurrencyPoint.from_dict(item) for item in data["concurrency_envelope"]),
            assumptions=tuple(str(item) for item in data.get("assumptions", ())),
            warnings=tuple(str(item) for item in data.get("warnings", ())),
            evidence=tuple(EvidenceRecord.from_dict(item) for item in data.get("evidence", ())),
        )
