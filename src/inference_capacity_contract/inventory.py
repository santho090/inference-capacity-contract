"""Caller-supplied provider, runtime, and measurement inventories."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Any

from .models import ContractError, HardwareSpec, RuntimeVariant
from .recipe import LLMDRoutingSpec, MeasuredGroupProfile

_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _optional_count(name: str, value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _optional_price(name: str, value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{name} must be a non-negative number")
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ContractError(f"{name} must be a finite non-negative number")
    return number


def _utilization(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError("memory_utilization_limit must be a number in (0, 1]")
    number = float(value)
    if not isfinite(number) or not 0 < number <= 1:
        raise ContractError("memory_utilization_limit must be a finite number in (0, 1]")
    return number


def _timestamp(name: str, value: object) -> str:
    text = _text(name, value)
    if not _RFC3339.fullmatch(text):
        raise ContractError(f"{name} must be an RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{name} must include a UTC offset")
    return text


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    return value


def _items(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{name} must be an array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class ProviderInstanceSpec:
    """One instance shape and its point-in-time price and availability."""

    provider_id: str
    region: str
    instance_type: str
    accelerator_id: str
    vendor: str
    devices_per_instance: int
    memory_bytes_per_device: int
    memory_utilization_limit: float = 0.9
    price_per_instance_hour: float | None = None
    price_currency: str | None = None
    max_instances: int | None = None
    interconnect: str | None = None
    source: str = "provider-inventory://caller-supplied"
    collected_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider_id", "region", "instance_type", "accelerator_id", "vendor", "source"):
            _text(name, getattr(self, name))
        _positive_int("devices_per_instance", self.devices_per_instance)
        _positive_int("memory_bytes_per_device", self.memory_bytes_per_device)
        object.__setattr__(self, "memory_utilization_limit", _utilization(self.memory_utilization_limit))
        object.__setattr__(
            self,
            "price_per_instance_hour",
            _optional_price("price_per_instance_hour", self.price_per_instance_hour),
        )
        _optional_count("max_instances", self.max_instances)
        currency = None if self.price_currency is None else _text("price_currency", self.price_currency)
        object.__setattr__(self, "price_currency", currency)
        if self.price_per_instance_hour is None and self.price_currency is not None:
            raise ContractError("price_currency requires price_per_instance_hour")
        if self.price_per_instance_hour is not None:
            if self.price_currency is None:
                raise ContractError("price_currency is required with price_per_instance_hour")
            if len(self.price_currency) != 3 or not self.price_currency.isalpha() or not self.price_currency.isupper():
                raise ContractError("price_currency must be a three-letter uppercase currency code")
        if (self.price_per_instance_hour is not None or self.max_instances is not None) and self.collected_at is None:
            raise ContractError("collected_at is required with provider price or availability")
        if self.collected_at is not None:
            _timestamp("collected_at", self.collected_at)
        if self.interconnect is not None:
            _text("interconnect", self.interconnect)

    @property
    def inventory_key(self) -> str:
        return f"{self.provider_id}/{self.region}/{self.instance_type}"

    def hardware_slice(self, device_count: int) -> HardwareSpec:
        _positive_int("device_count", device_count)
        if device_count > self.devices_per_instance:
            raise ContractError("runtime requires more devices than the provider instance contains")
        device_price = (
            None if self.price_per_instance_hour is None else self.price_per_instance_hour / self.devices_per_instance
        )
        return HardwareSpec(
            hardware_id=f"{self.inventory_key}#devices={device_count}",
            vendor=self.vendor,
            device_count=device_count,
            memory_bytes_per_device=self.memory_bytes_per_device,
            memory_utilization_limit=self.memory_utilization_limit,
            price_per_device_hour=device_price,
            interconnect=self.interconnect,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "region": self.region,
            "instance_type": self.instance_type,
            "accelerator_id": self.accelerator_id,
            "vendor": self.vendor,
            "devices_per_instance": self.devices_per_instance,
            "memory_bytes_per_device": self.memory_bytes_per_device,
            "memory_utilization_limit": self.memory_utilization_limit,
            "price_per_instance_hour": self.price_per_instance_hour,
            "price_currency": self.price_currency,
            "max_instances": self.max_instances,
            "interconnect": self.interconnect,
            "source": self.source,
            "collected_at": self.collected_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderInstanceSpec:
        return cls(
            provider_id=_text("provider_id", data.get("provider_id")),
            region=_text("region", data.get("region")),
            instance_type=_text("instance_type", data.get("instance_type")),
            accelerator_id=_text("accelerator_id", data.get("accelerator_id")),
            vendor=_text("vendor", data.get("vendor")),
            devices_per_instance=_positive_int("devices_per_instance", data.get("devices_per_instance")),
            memory_bytes_per_device=_positive_int("memory_bytes_per_device", data.get("memory_bytes_per_device")),
            memory_utilization_limit=_utilization(data.get("memory_utilization_limit", 0.9)),
            price_per_instance_hour=_optional_price("price_per_instance_hour", data.get("price_per_instance_hour")),
            price_currency=(
                None if data.get("price_currency") is None else _text("price_currency", data["price_currency"])
            ),
            max_instances=_optional_count("max_instances", data.get("max_instances")),
            interconnect=(None if data.get("interconnect") is None else _text("interconnect", data["interconnect"])),
            source=_text("source", data.get("source", "provider-inventory://caller-supplied")),
            collected_at=(
                None if data.get("collected_at") is None else _timestamp("collected_at", data["collected_at"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ProviderInventory:
    schema_version: str
    items: tuple[ProviderInstanceSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "provider-inventory-1.0":
            raise ContractError("unsupported provider inventory schema_version")
        if not self.items:
            raise ContractError("provider inventory cannot be empty")
        keys = tuple(item.inventory_key for item in self.items)
        if len(keys) != len(set(keys)):
            raise ContractError("provider inventory contains duplicate provider/region/instance entries")
        object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProviderInventory:
        items = _items("provider inventory items", data.get("items"))
        return cls(
            _text("schema_version", data.get("schema_version")),
            tuple(ProviderInstanceSpec.from_dict(_mapping("provider inventory item", item)) for item in items),
        )


@dataclass(frozen=True, slots=True)
class RuntimeOption:
    runtime_id: str
    runtime: RuntimeVariant
    source: str
    expert_parallel_size: int = 1
    routing: LLMDRoutingSpec = LLMDRoutingSpec()
    supported_accelerator_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text("runtime_id", self.runtime_id)
        _text("source", self.source)
        if self.runtime.max_num_seqs is None:
            raise ContractError("runtime options require max_num_seqs")
        _positive_int("expert_parallel_size", self.expert_parallel_size)
        if self.runtime.tensor_parallel_size % self.expert_parallel_size:
            raise ContractError("expert_parallel_size must divide tensor_parallel_size")
        if self.expert_parallel_size > 1 and self.runtime.weight_bytes_per_device_override is None:
            raise ContractError("expert-parallel runtime options require weight_bytes_per_device_override")
        accelerators = tuple(_text("supported_accelerator_ids item", item) for item in self.supported_accelerator_ids)
        if len(accelerators) != len(set(accelerators)):
            raise ContractError("supported_accelerator_ids contains duplicates")
        object.__setattr__(self, "supported_accelerator_ids", accelerators)

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "runtime": self.runtime.to_dict(),
            "source": self.source,
            "expert_parallel_size": self.expert_parallel_size,
            "routing": self.routing.to_dict(),
            "supported_accelerator_ids": list(self.supported_accelerator_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeOption:
        accelerators = _items("supported_accelerator_ids", data.get("supported_accelerator_ids", ()))
        return cls(
            _text("runtime_id", data.get("runtime_id")),
            RuntimeVariant.from_dict(_mapping("runtime", data.get("runtime"))),
            _text("source", data.get("source")),
            _positive_int("expert_parallel_size", data.get("expert_parallel_size", 1)),
            LLMDRoutingSpec.from_dict(_mapping("routing", data.get("routing", {}))),
            tuple(_text("supported_accelerator_ids item", item) for item in accelerators),
        )


@dataclass(frozen=True, slots=True)
class RuntimeInventory:
    schema_version: str
    items: tuple[RuntimeOption, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "runtime-inventory-1.0":
            raise ContractError("unsupported runtime inventory schema_version")
        if not self.items:
            raise ContractError("runtime inventory cannot be empty")
        ids = tuple(item.runtime_id for item in self.items)
        if len(ids) != len(set(ids)):
            raise ContractError("runtime inventory contains duplicate runtime_id values")
        object.__setattr__(self, "items", tuple(self.items))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeInventory:
        items = _items("runtime inventory items", data.get("items"))
        return cls(
            _text("schema_version", data.get("schema_version")),
            tuple(RuntimeOption.from_dict(_mapping("runtime inventory item", item)) for item in items),
        )


@dataclass(frozen=True, slots=True)
class MeasurementInventory:
    schema_version: str
    items: tuple[MeasuredGroupProfile, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "measurement-inventory-1.0":
            raise ContractError("unsupported measurement inventory schema_version")
        profile_ids = tuple(item.profile_id for item in self.items)
        if len(profile_ids) != len(set(profile_ids)):
            raise ContractError("measurement inventory contains duplicate profile_id values")
        object.__setattr__(self, "items", tuple(self.items))

    def for_recipe(
        self, fingerprint: str, context_tokens: int, input_tokens: float, output_tokens: float
    ) -> MeasuredGroupProfile | None:
        matches = tuple(
            item
            for item in self.items
            if item.recipe_variant_fingerprint == fingerprint
            and item.context_tokens == context_tokens
            and (not input_tokens or item.input_tokens_per_request == input_tokens)
            and (not output_tokens or item.output_tokens_per_request == output_tokens)
        )
        if len(matches) > 1:
            raise ContractError("measurement inventory has more than one matching operating point")
        return matches[0] if matches else None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "items": [item.to_dict() for item in self.items]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeasurementInventory:
        items = _items("measurement inventory items", data.get("items", ()))
        return cls(
            _text("schema_version", data.get("schema_version")),
            tuple(MeasuredGroupProfile.from_dict(_mapping("measurement inventory item", item)) for item in items),
        )
