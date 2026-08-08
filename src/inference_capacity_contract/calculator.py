"""Deterministic memory and KV-cache feasibility calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .models import (
    WEIGHT_BYTES,
    CapacityContract,
    ContractError,
    HardwareInventory,
    HardwareSpec,
    ModelSpec,
    RuntimeVariant,
    ValidationLevel,
)


BYTES_PER_GIB = 1024**3


def _kv_bytes_per_token(model: ModelSpec, runtime: RuntimeVariant) -> int:
    if model.kv_bytes_per_token_override is not None:
        return model.kv_bytes_per_token_override
    if model.attention_type in {"mla", "custom"}:
        raise ContractError("MLA/custom attention requires kv_bytes_per_token_override in v0")
    return int(2 * model.num_layers * model.num_kv_heads * model.head_dim * WEIGHT_BYTES[runtime.kv_cache_dtype])


def capacity_for(model: ModelSpec, hardware: HardwareSpec, runtime: RuntimeVariant) -> CapacityContract:
    """Calculate a single replica's analytical capacity contract.

    The result is a lower-bound feasibility calculation. It does not predict
    TTFT, TPOT, throughput, or a safe autoscaling policy.
    """

    warnings: list[str] = []
    assumptions: list[str] = [
        "Weights are divided evenly across tensor/data-parallel devices.",
        "KV cache uses paged storage with the declared bytes per token.",
        "Activation and runtime reserves are explicit inputs; no runtime profiling is performed.",
        "Analytical fit is not runtime initialization or SLO validation.",
    ]
    if hardware.vendor.lower() not in {vendor.lower() for vendor in runtime.supported_vendors}:
        warnings.append(f"runtime {runtime.engine} does not declare support for hardware vendor {hardware.vendor}")
    if hardware.device_count != runtime.devices_required:
        warnings.append(
            f"hardware device_count={hardware.device_count} does not equal runtime devices_required={runtime.devices_required}"
        )

    usable_bytes = int(hardware.memory_bytes_per_device * hardware.memory_utilization_limit)
    weights_per_device = (model.weight_bytes + runtime.tensor_parallel_size - 1) // runtime.tensor_parallel_size
    runtime_reserve = runtime.runtime_overhead_bytes_per_device
    activation_reserve = runtime.activation_reserve_bytes_per_device
    kv_available = usable_bytes - weights_per_device - runtime_reserve - activation_reserve
    kv_bytes = _kv_bytes_per_token(model, runtime)
    memory_fit = (
        hardware.vendor.lower() in {vendor.lower() for vendor in runtime.supported_vendors}
        and hardware.device_count == runtime.devices_required
        and kv_available >= 0
    )
    if kv_available < 0:
        warnings.append("weights and declared runtime reserves exceed the usable memory budget")
    if model.max_model_len is None:
        warnings.append("model max_model_len is unknown; max_context_tokens is bounded only by available KV memory")
    if runtime.runtime_overhead_bytes_per_device == 0:
        warnings.append("runtime overhead reserve is zero; provide a measured reserve before treating fit as production-safe")
    if runtime.activation_reserve_bytes_per_device == 0:
        warnings.append("activation reserve is zero; provide a measured reserve before treating fit as production-safe")

    kv_capacity_tokens = max(0, kv_available // kv_bytes)
    fits = memory_fit and kv_capacity_tokens > 0
    if memory_fit and kv_capacity_tokens == 0:
        warnings.append("usable memory has no room for one KV-cache token")
    max_context_tokens = kv_capacity_tokens if model.max_model_len is None else min(model.max_model_len, kv_capacity_tokens)
    per_engine_sequences = max_context_tokens // runtime.block_size_tokens
    if runtime.max_num_seqs is not None:
        per_engine_sequences = min(per_engine_sequences, runtime.max_num_seqs)
    max_sequences = max(0, per_engine_sequences * runtime.data_parallel_size) if fits else 0
    remaining = max(0, kv_available)
    return CapacityContract(
        schema_version="capacity-contract-1.0",
        model=model,
        hardware=hardware,
        runtime=runtime,
        fits=fits,
        validation_level=ValidationLevel.ANALYTICALLY_FEASIBLE,
        memory_bytes_per_device={
            "usable_budget": usable_bytes,
            "weights": weights_per_device,
            "runtime_reserve": runtime_reserve,
            "activation_reserve": activation_reserve,
            "kv_available": max(0, kv_available),
        },
        kv_bytes_per_token=kv_bytes,
        kv_capacity_tokens=kv_capacity_tokens,
        max_context_tokens=max_context_tokens,
        max_concurrent_sequences=max_sequences,
        remaining_bytes_per_device=remaining,
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
    )


@dataclass(frozen=True, slots=True)
class FitCandidate:
    hardware_id: str
    fits: bool
    max_context_tokens: int
    max_concurrent_sequences: int
    remaining_bytes_per_device: int
    warnings: tuple[str, ...]
    contract: CapacityContract

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "fits": self.fits,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_sequences": self.max_concurrent_sequences,
            "remaining_bytes_per_device": self.remaining_bytes_per_device,
            "warnings": list(self.warnings),
            "contract": self.contract.to_dict(),
        }


def what_fits(
    model: ModelSpec,
    inventory: HardwareInventory,
    runtimes: RuntimeVariant | Iterable[RuntimeVariant],
) -> list[FitCandidate]:
    """Return deterministic reverse-fit candidates, sorted by hardware and runtime."""

    runtime_list = [runtimes] if isinstance(runtimes, RuntimeVariant) else list(runtimes)
    if not runtime_list:
        raise ContractError("at least one runtime is required")
    candidates: list[FitCandidate] = []
    for hardware in inventory.items:
        for runtime in runtime_list:
            contract = capacity_for(model, hardware, runtime)
            candidates.append(
                FitCandidate(
                    hardware_id=hardware.hardware_id,
                    fits=contract.fits,
                    max_context_tokens=contract.max_context_tokens,
                    max_concurrent_sequences=contract.max_concurrent_sequences,
                    remaining_bytes_per_device=contract.remaining_bytes_per_device,
                    warnings=contract.warnings,
                    contract=contract,
                )
            )
    return sorted(candidates, key=lambda item: (not item.fits, item.hardware_id, item.contract.runtime.engine))
