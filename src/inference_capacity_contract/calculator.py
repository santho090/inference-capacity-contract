"""Deterministic memory and KV-cache feasibility calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .arithmetic import ceil_div, sequence_capacity
from .models import (
    DTYPE_BITS,
    CapacityContract,
    ConcurrencyPoint,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareInventory,
    HardwareSpec,
    KVCapacityMode,
    ModelSpec,
    RuntimeVariant,
    ValidationLevel,
)


def _kv_bytes_per_token_per_device(model: ModelSpec, runtime: RuntimeVariant) -> int:
    override = model.kv_bytes_per_token_per_device_override
    if model.attention_type in {"mla", "hybrid", "custom"} or DTYPE_BITS[runtime.kv_cache_dtype] < 8:
        raise ContractError("MLA, hybrid, custom, and sub-byte KV layouts require a runtime KV capacity envelope")
    if override is not None:
        return override
    if runtime.tensor_parallel_size == 1:
        kv_heads_per_device = model.num_kv_heads
    elif runtime.kv_heads_per_device is None:
        raise ContractError("tensor-parallel KV layout is runtime-specific; provide kv_heads_per_device")
    else:
        kv_heads_per_device = runtime.kv_heads_per_device
    if kv_heads_per_device > model.num_kv_heads:
        raise ContractError("kv_heads_per_device cannot exceed model num_kv_heads")
    if runtime.tensor_parallel_size * kv_heads_per_device < model.num_kv_heads:
        raise ContractError("kv_heads_per_device under-represents the model KV layout across tensor-parallel devices")
    kv_elements_per_token = 2 * model.num_layers * kv_heads_per_device * model.head_dim
    return ceil_div(kv_elements_per_token * DTYPE_BITS[runtime.kv_cache_dtype], 8)


def _context_points(max_context_tokens: int, block_size_tokens: int) -> tuple[int, ...]:
    candidates = {
        block_size_tokens,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        max_context_tokens,
    }
    return tuple(sorted(point for point in candidates if 0 < point <= max_context_tokens))


def capacity_for(
    model: ModelSpec,
    hardware: HardwareSpec,
    runtime: RuntimeVariant,
    *,
    context_points: Iterable[int] | None = None,
) -> CapacityContract:
    """Calculate one tensor-parallel replica's analytical capacity contract."""

    warnings: list[str] = []
    assumptions: list[str] = [
        "KV cache uses paged storage with the declared block size.",
        "KV bytes are calculated per device using the declared runtime KV layout.",
        "Activation and runtime reserves are explicit inputs; no runtime profiling is performed.",
        "Analytical fit is not runtime initialization or SLO validation.",
    ]
    supported_vendors = {vendor.lower() for vendor in runtime.supported_vendors}
    vendor_supported = hardware.vendor.lower() in supported_vendors
    topology_supported = hardware.device_count == runtime.devices_required
    if not vendor_supported:
        warnings.append(f"runtime {runtime.engine} does not declare support for hardware vendor {hardware.vendor}")
    if not topology_supported:
        warnings.append(
            f"hardware device_count={hardware.device_count} does not equal "
            f"tensor_parallel_size={runtime.tensor_parallel_size}"
        )

    usable_bytes = runtime.memory_budget_bytes(hardware)
    if runtime.weight_bytes_per_device_override is not None:
        weights_per_device = runtime.weight_bytes_per_device_override
        assumptions.append("Per-device resident weight bytes use the explicit runtime override.")
    else:
        weights_per_device = ceil_div(model.weight_bytes, runtime.tensor_parallel_size)
        assumptions.append("Weights are divided evenly across tensor-parallel devices.")
        if runtime.tensor_parallel_size > 1:
            warnings.append(
                "tensor-parallel weight bytes assume even sharding with no replicated tensors or metadata; "
                "provide weight_bytes_per_device_override for an exact layout"
            )
    runtime_reserve = runtime.runtime_overhead_bytes_per_device
    activation_reserve = runtime.activation_reserve_bytes_per_device
    kv_available = usable_bytes - weights_per_device - runtime_reserve - activation_reserve
    requested_points: tuple[int, ...] | None = None
    if context_points is not None:
        requested_points = tuple(sorted(set(int(point) for point in context_points)))
        if not requested_points:
            raise ContractError("context_points cannot be empty")
        if any(point <= 0 for point in requested_points):
            raise ContractError("context points must be positive")

    runtime_envelope = runtime.kv_capacity_envelope_override
    if runtime_envelope and model.kv_bytes_per_token_per_device_override is not None:
        raise ContractError("supply either a linear KV bytes/token override or a runtime KV capacity envelope")
    if kv_available < 0:
        warnings.append("weights and declared runtime reserves exceed the usable memory budget")
    if model.max_model_len is None:
        warnings.append("model max_model_len is unknown; max_context_tokens is bounded only by KV memory")
    if runtime.runtime_overhead_bytes_per_device == 0:
        warnings.append("runtime overhead reserve is zero; provide a measured reserve before production use")
    if runtime.activation_reserve_bytes_per_device == 0:
        warnings.append("activation reserve is zero; provide a measured reserve before production use")

    envelope: list[ConcurrencyPoint] = []
    if runtime_envelope:
        capacity_mode = KVCapacityMode.CONTEXT_ENVELOPE
        kv_bytes_per_token = None
        kv_capacity_blocks = None
        kv_capacity_tokens = None
        exact_hardware = runtime.kv_capacity_hardware_id == hardware.hardware_id
        exact_memory = runtime.kv_capacity_memory_bytes_per_device == max(0, kv_available)
        if not exact_hardware:
            warnings.append("runtime KV capacity envelope was measured on a different hardware ID")
        if not exact_memory:
            warnings.append("runtime KV capacity envelope was measured with a different KV memory budget")
        available_points = {
            point.context_tokens: point.max_sequences
            for point in runtime_envelope
            if model.max_model_len is None or point.context_tokens <= model.max_model_len
        }
        fits = vendor_supported and topology_supported and exact_hardware and exact_memory and bool(available_points)
        max_context_tokens = 0
        if fits:
            max_context_tokens = max(available_points)
            if requested_points is None:
                sampled_points = tuple(available_points)
            else:
                missing = tuple(point for point in requested_points if point not in available_points)
                if missing:
                    raise ContractError(
                        "runtime KV capacity envelope has no exact point for context(s): "
                        + ", ".join(str(point) for point in missing)
                    )
                sampled_points = requested_points
            for context_tokens in sampled_points:
                envelope.append(
                    ConcurrencyPoint(
                        context_tokens,
                        ceil_div(context_tokens, runtime.block_size_tokens),
                        available_points[context_tokens],
                    )
                )
        assumptions.append(
            "KV sequence capacity uses exact context-bound runtime observations; no interpolation is used."
        )
    else:
        capacity_mode = KVCapacityMode.LINEAR
        kv_bytes_per_token = _kv_bytes_per_token_per_device(model, runtime)
        kv_block_bytes = kv_bytes_per_token * runtime.block_size_tokens
        if 0 <= kv_available < kv_block_bytes:
            warnings.append("usable memory has no room for one KV-cache block")
        kv_capacity_blocks = max(0, kv_available // kv_block_bytes)
        kv_capacity_tokens = kv_capacity_blocks * runtime.block_size_tokens
        fits = vendor_supported and topology_supported and kv_capacity_blocks > 0
        max_context_tokens = kv_capacity_tokens
        if model.max_model_len is not None:
            max_context_tokens = min(max_context_tokens, model.max_model_len)
            if fits and kv_capacity_tokens < model.max_model_len:
                warnings.append("KV memory cannot hold the model's declared maximum context")
        if not fits:
            max_context_tokens = 0
        if requested_points is None:
            sampled_points = _context_points(max_context_tokens, runtime.block_size_tokens)
        else:
            sampled_points = requested_points
            if fits:
                if any(point > max_context_tokens for point in sampled_points):
                    raise ContractError("context points cannot exceed max_context_tokens")
            else:
                sampled_points = ()
        for context_tokens in sampled_points:
            blocks_per_sequence, max_sequences = sequence_capacity(
                kv_capacity_blocks,
                runtime.block_size_tokens,
                context_tokens,
                runtime.max_num_seqs,
            )
            envelope.append(ConcurrencyPoint(context_tokens, blocks_per_sequence, max_sequences))

    return CapacityContract(
        schema_version="capacity-contract-3.0",
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
        kv_capacity_mode=capacity_mode,
        kv_bytes_per_token_per_device=kv_bytes_per_token,
        kv_block_size_tokens=runtime.block_size_tokens,
        kv_capacity_blocks_per_device=kv_capacity_blocks,
        kv_capacity_tokens_per_device=kv_capacity_tokens,
        max_context_tokens=max_context_tokens,
        concurrency_envelope=tuple(envelope),
        assumptions=tuple(assumptions),
        warnings=tuple(warnings),
        evidence=(
            EvidenceRecord(
                evidence_id="analytical-capacity-contract-3.0",
                kind=EvidenceKind.ANALYTICAL,
                source="urn:inference-capacity-contract:calculator:capacity-contract-3.0",
                scope=(
                    f"{model.model_id}@{model.revision} on {hardware.hardware_id}/{runtime.engine}-{runtime.version}"
                ),
                metrics={
                    "formula_version": "capacity-contract-3.0",
                    "validation_level": "analytically-feasible",
                },
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class FitCandidate:
    hardware_id: str
    runtime_engine: str
    fits: bool
    max_context_tokens: int
    remaining_bytes_per_device: int
    warnings: tuple[str, ...]
    unsupported_reason: str | None
    contract: CapacityContract | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hardware_id": self.hardware_id,
            "runtime_engine": self.runtime_engine,
            "fits": self.fits,
            "max_context_tokens": self.max_context_tokens,
            "remaining_bytes_per_device": self.remaining_bytes_per_device,
            "warnings": list(self.warnings),
            "unsupported_reason": self.unsupported_reason,
            "contract": None if self.contract is None else self.contract.to_dict(),
        }


def what_fits(
    model: ModelSpec,
    inventory: HardwareInventory,
    runtimes: RuntimeVariant | Iterable[RuntimeVariant],
) -> list[FitCandidate]:
    """Evaluate every inventory/runtime pair without aborting on one unsupported candidate."""

    runtime_list = [runtimes] if isinstance(runtimes, RuntimeVariant) else list(runtimes)
    if not runtime_list:
        raise ContractError("at least one runtime is required")
    candidates: list[FitCandidate] = []
    for hardware in inventory.items:
        for runtime in runtime_list:
            try:
                contract = capacity_for(model, hardware, runtime)
            except ContractError as exc:
                candidates.append(
                    FitCandidate(
                        hardware_id=hardware.hardware_id,
                        runtime_engine=runtime.engine,
                        fits=False,
                        max_context_tokens=0,
                        remaining_bytes_per_device=0,
                        warnings=(),
                        unsupported_reason=str(exc),
                        contract=None,
                    )
                )
                continue
            candidates.append(
                FitCandidate(
                    hardware_id=hardware.hardware_id,
                    runtime_engine=runtime.engine,
                    fits=contract.fits,
                    max_context_tokens=contract.max_context_tokens,
                    remaining_bytes_per_device=contract.memory_bytes_per_device["kv_available"],
                    warnings=contract.warnings,
                    unsupported_reason=None,
                    contract=contract,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (not item.fits, item.hardware_id, item.runtime_engine),
    )
