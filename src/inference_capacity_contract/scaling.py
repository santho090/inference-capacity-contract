"""Build replica recommendations from measured workload data."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from .models import CapacityContract, ContractError, EvidenceKind, WorkloadProfile


def _required(
    demand: float,
    capacity: float | None,
    target_utilization: float,
) -> int | None:
    if demand <= 0:
        return 0
    if capacity is None or capacity <= 0:
        return None
    return max(1, ceil(demand / (capacity * target_utilization)))


@dataclass(frozen=True, slots=True)
class ScalingRecommendation:
    """A replica recommendation calculated from a measured profile."""

    schema_version: str
    contract: CapacityContract
    profile: WorkloadProfile
    required_replicas: int
    recommended_replicas: int
    drivers: dict[str, int]
    constrained_by: str
    within_bounds: bool
    estimated_gpu_hours_per_hour: float
    estimated_gpu_hour_savings_vs_baseline: float | None
    estimated_hourly_cost: float | None
    estimated_hourly_savings_vs_baseline: float | None
    suggestions: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.to_dict(),
            "profile": self.profile.to_dict(),
            "required_replicas": self.required_replicas,
            "recommended_replicas": self.recommended_replicas,
            "drivers": dict(self.drivers),
            "constrained_by": self.constrained_by,
            "within_bounds": self.within_bounds,
            "estimated_gpu_hours_per_hour": self.estimated_gpu_hours_per_hour,
            "estimated_gpu_hour_savings_vs_baseline": (self.estimated_gpu_hour_savings_vs_baseline),
            "estimated_hourly_cost": self.estimated_hourly_cost,
            "estimated_hourly_savings_vs_baseline": (self.estimated_hourly_savings_vs_baseline),
            "suggestions": list(self.suggestions),
            "warnings": list(self.warnings),
        }


def recommend_scale(contract: CapacityContract, profile: WorkloadProfile) -> ScalingRecommendation:
    """Calculate a measured-profile replica recommendation.

    Static memory fit alone is not enough. The profile must have a measured
    evidence record and must match the contract's identity. The result is a
    planning input and does not change Kubernetes or cloud resources.
    """

    if not contract.fits:
        raise ContractError("cannot recommend replicas for a contract that does not fit analytically")
    if not any(record.kind == EvidenceKind.MEASURED for record in profile.evidence):
        raise ContractError("autoscale recommendations require at least one measured evidence record")
    if profile.model_id != contract.model.model_id or profile.model_revision != contract.model.revision:
        raise ContractError("workload profile model identity does not match the capacity contract")
    if profile.hardware_id != contract.hardware.hardware_id:
        raise ContractError("workload profile hardware identity does not match the capacity contract")
    if profile.runtime_engine != contract.runtime.engine or profile.runtime_version != contract.runtime.version:
        raise ContractError("workload profile runtime identity does not match the capacity contract")

    demand = {
        "requests_per_second": profile.request_rate_per_second,
        "prefill_tokens_per_second": profile.request_rate_per_second * profile.input_tokens_per_request,
        "decode_tokens_per_second": profile.request_rate_per_second * profile.output_tokens_per_request,
    }
    capacities = {
        "requests_per_second": profile.sustainable_requests_per_replica_per_second,
        "prefill_tokens_per_second": profile.sustainable_prefill_tokens_per_replica_per_second,
        "decode_tokens_per_second": profile.sustainable_decode_tokens_per_replica_per_second,
    }
    drivers: dict[str, int] = {}
    unmeasured_drivers: list[str] = []
    for name, value in demand.items():
        if value <= 0:
            continue
        required = _required(value, capacities[name], profile.target_utilization)
        if required is not None:
            drivers[name] = required
        else:
            unmeasured_drivers.append(name)
    if profile.peak_concurrent_sequences is not None:
        if profile.concurrency_context_tokens is None:
            raise ContractError("concurrency_context_tokens is required with peak_concurrent_sequences")
        if profile.sustainable_concurrent_sequences_per_replica is None:
            raise ContractError(
                "sustainable_concurrent_sequences_per_replica is required with peak_concurrent_sequences"
            )
        if profile.concurrency_context_tokens > contract.max_context_tokens:
            raise ContractError(
                "concurrency_context_tokens exceeds the contract's supported context; "
                "choose another candidate or reduce the context requirement"
            )
        analytical_concurrency_limit = contract.max_sequences_at(profile.concurrency_context_tokens)
        sustainable_concurrency = min(
            profile.sustainable_concurrent_sequences_per_replica,
            analytical_concurrency_limit,
        )
        concurrency_required = _required(
            profile.peak_concurrent_sequences,
            float(sustainable_concurrency),
            profile.target_utilization,
        )
        if concurrency_required is None:
            unmeasured_drivers.append("peak_concurrent_sequences")
        else:
            drivers["peak_concurrent_sequences"] = concurrency_required

    required_replicas = max(drivers.values(), default=0)
    buffered_replicas = ceil(required_replicas * profile.scale_up_buffer)
    recommended = max(profile.min_replicas, buffered_replicas)
    warnings: list[str] = []
    if profile.max_replicas is not None and recommended > profile.max_replicas:
        warnings.append("max_replicas is below the measured demand plus configured buffer")
        recommended = profile.max_replicas
    within_bounds = not unmeasured_drivers and recommended >= required_replicas
    if unmeasured_drivers:
        warnings.extend(f"{name} demand has no measured per-replica capacity" for name in unmeasured_drivers)
        warnings.append("recommendation is incomplete until every non-zero demand driver is measured")
    if (
        profile.peak_concurrent_sequences is not None
        and profile.sustainable_concurrent_sequences_per_replica is not None
        and profile.concurrency_context_tokens is not None
        and profile.sustainable_concurrent_sequences_per_replica
        > contract.max_sequences_at(profile.concurrency_context_tokens)
    ):
        warnings.append("measured concurrency exceeds the analytical KV bound; the analytical bound was used")
    if not within_bounds:
        warnings.append("recommended replica count cannot satisfy measured demand within configured bounds")
    constrained_by = max(drivers.items(), key=lambda item: item[1])[0] if drivers else "min-replicas-floor"

    gpu_hours = float(recommended * contract.hardware.device_count)
    gpu_hour_savings = None
    if profile.baseline_replicas is not None:
        gpu_hour_savings = float((profile.baseline_replicas - recommended) * contract.hardware.device_count)
    cost_per_replica_hour: float | None = None
    if contract.hardware.price_per_device_hour is not None:
        cost_per_replica_hour = contract.hardware.price_per_device_hour * contract.hardware.device_count
    estimated_cost = None if cost_per_replica_hour is None else recommended * cost_per_replica_hour
    savings = None
    if profile.baseline_replicas is not None and cost_per_replica_hour is not None:
        savings = (profile.baseline_replicas - recommended) * cost_per_replica_hour
    if cost_per_replica_hour is None:
        warnings.append("hardware price is unknown; hourly cost savings are not estimated")

    suggestions: list[str] = []
    if profile.baseline_replicas is not None and recommended < profile.baseline_replicas:
        suggestions.append(
            f"test reducing replicas from {profile.baseline_replicas} to {recommended} under the measured profile"
        )
    elif profile.baseline_replicas is not None and recommended > profile.baseline_replicas:
        suggestions.append(
            f"test increasing replicas from {profile.baseline_replicas} to {recommended} before accepting queue growth"
        )
    else:
        suggestions.append("use the recommendation as a policy input and validate it against a live canary")
    if drivers:
        suggestions.append(f"collect a new measured profile when the {constrained_by} driver changes materially")
    else:
        suggestions.append("review min_replicas before reducing the idle floor")

    return ScalingRecommendation(
        schema_version="scaling-recommendation-2.0",
        contract=contract,
        profile=profile,
        required_replicas=required_replicas,
        recommended_replicas=recommended,
        drivers=drivers,
        constrained_by=constrained_by,
        within_bounds=within_bounds,
        estimated_gpu_hours_per_hour=gpu_hours,
        estimated_gpu_hour_savings_vs_baseline=gpu_hour_savings,
        estimated_hourly_cost=estimated_cost,
        estimated_hourly_savings_vs_baseline=savings,
        suggestions=tuple(suggestions),
        warnings=tuple(warnings),
    )
