"""Public domain objects for the capacity-contract format.

The objects are deliberately descriptive. They do not start servers, inspect
GPUs, or issue scaling actions. Every calculation is tied to an immutable
model/runtime/hardware tuple and records assumptions in the returned contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Architecture and weight facts for one immutable model artifact."""

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
    kv_bytes_per_token_override: int | None = None
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
        if self.kv_bytes_per_token_override is not None:
            _positive("kv_bytes_per_token_override", self.kv_bytes_per_token_override)
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
            "kv_bytes_per_token_override": self.kv_bytes_per_token_override,
            "architecture": self.architecture,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True, slots=True)
class HardwareSpec:
    """One serving-replica hardware topology."""

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
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareSpec":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})


@dataclass(frozen=True, slots=True)
class RuntimeVariant:
    """Pinned runtime configuration for one serving replica."""

    engine: str
    version: str
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    kv_cache_dtype: str = "bf16"
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
        for name in ("tensor_parallel_size", "data_parallel_size", "block_size_tokens"):
            _positive(name, int(getattr(self, name)))
        for name in ("runtime_overhead_bytes_per_device", "activation_reserve_bytes_per_device"):
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
        return self.tensor_parallel_size * self.data_parallel_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "version": self.version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "kv_cache_dtype": self.kv_cache_dtype,
            "runtime_overhead_bytes_per_device": self.runtime_overhead_bytes_per_device,
            "activation_reserve_bytes_per_device": self.activation_reserve_bytes_per_device,
            "max_num_seqs": self.max_num_seqs,
            "block_size_tokens": self.block_size_tokens,
            "supported_vendors": list(self.supported_vendors),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeVariant":
        values = dict(data)
        for key in ("supported_vendors", "notes"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


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
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceRecord":
        values = dict(data)
        values["kind"] = EvidenceKind(values["kind"])
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


@dataclass(frozen=True, slots=True)
class WorkloadProfile:
    """Measured demand and sustainable per-replica capacity for one variant.

    This is deliberately an average-rate profile rather than a queueing model.
    It is sufficient for a transparent first autoscale recommendation, but not
    a replacement for a runtime-specific SLO/load model.
    """

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
    peak_concurrent_sequences: int | None = None
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
        for name in ("request_rate_per_second", "input_tokens_per_request", "output_tokens_per_request"):
            value = float(getattr(self, name))
            if value < 0:
                raise ContractError(f"{name} cannot be negative")
        sustainable_fields = (
            "sustainable_requests_per_replica_per_second",
            "sustainable_prefill_tokens_per_replica_per_second",
            "sustainable_decode_tokens_per_replica_per_second",
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
            "sustainable_requests_per_replica_per_second": self.sustainable_requests_per_replica_per_second,
            "sustainable_prefill_tokens_per_replica_per_second": self.sustainable_prefill_tokens_per_replica_per_second,
            "sustainable_decode_tokens_per_replica_per_second": self.sustainable_decode_tokens_per_replica_per_second,
            "peak_concurrent_sequences": self.peak_concurrent_sequences,
            "target_utilization": self.target_utilization,
            "scale_up_buffer": self.scale_up_buffer,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "baseline_replicas": self.baseline_replicas,
            "evidence": [record.to_dict() for record in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkloadProfile":
        values = dict(data)
        values["evidence"] = tuple(EvidenceRecord.from_dict(item) for item in values.get("evidence", []))
        return cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})


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
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareInventory":
        return cls(tuple(HardwareSpec.from_dict(item) for item in data["items"]))


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
    kv_bytes_per_token: int
    kv_capacity_tokens: int
    max_context_tokens: int
    max_concurrent_sequences: int
    remaining_bytes_per_device: int
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "hardware": self.hardware.to_dict(),
            "runtime": self.runtime.to_dict(),
            "fits": self.fits,
            "validation_level": self.validation_level.value,
            "memory_bytes_per_device": dict(self.memory_bytes_per_device),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "remaining_bytes_per_device": self.remaining_bytes_per_device,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def with_evidence(self, *records: EvidenceRecord) -> "CapacityContract":
        """Return this contract with additional provenance records attached.

        Adding evidence does not silently promote ``validation_level``. A
        measured record can support a later runtime or SLO validation process,
        but the producer must explicitly perform that process before changing
        the level. This keeps analytical output from being presented as a
        runtime guarantee.
        """

        if not records:
            return self
        return replace(self, evidence=self.evidence + tuple(records))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapacityContract":
        return cls(
            schema_version=data["schema_version"],
            model=ModelSpec.from_dict(data["model"]),
            hardware=HardwareSpec.from_dict(data["hardware"]),
            runtime=RuntimeVariant.from_dict(data["runtime"]),
            fits=bool(data["fits"]),
            validation_level=ValidationLevel(data["validation_level"]),
            memory_bytes_per_device=data["memory_bytes_per_device"],
            kv_bytes_per_token=int(data["kv_bytes_per_token"]),
            kv_capacity_tokens=int(data["kv_capacity_tokens"]),
            max_context_tokens=int(data["max_context_tokens"]),
            max_concurrent_sequences=int(data["max_concurrent_sequences"]),
            remaining_bytes_per_device=int(data["remaining_bytes_per_device"]),
            assumptions=tuple(data.get("assumptions", [])),
            warnings=tuple(data.get("warnings", [])),
            evidence=tuple(EvidenceRecord.from_dict(item) for item in data.get("evidence", [])),
        )
