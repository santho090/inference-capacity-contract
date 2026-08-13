"""Single-model planning across caller-supplied provider inventory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .arithmetic import ceil_div
from .calculator import capacity_for
from .importers import ModelManifest
from .inventory import (
    MeasurementInventory,
    ProviderInstanceSpec,
    ProviderInventory,
    RuntimeInventory,
    RuntimeOption,
)
from .models import CapacityContract, ContractError, EvidenceKind, EvidenceRecord, ModelSpec
from .recipe import AuditStatus, LoadRequirement, ParallelTopology, RecipeAudit, ServingRecipe, audit_recipe


class CandidateStatus(StrEnum):
    FITS = "fits"
    FEASIBLE = "feasible"
    INCOMPLETE = "incomplete"
    INSUFFICIENT_INVENTORY = "insufficient-inventory"
    REJECTED = "rejected"


class PlanningObjective(StrEnum):
    LOWEST_COST = "lowest-cost"
    FEWEST_ALLOCATED_DEVICES = "fewest-allocated-devices"


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    serving_replicas: int
    required_instances: int
    serving_devices: int
    allocated_devices: int
    idle_devices: int
    devices_per_replica: int
    replicas_per_instance: int
    available_instances: int | None
    available_replicas: int | None
    fits_available_inventory: bool | None

    def __post_init__(self) -> None:
        if self.serving_replicas <= 0 or self.required_instances <= 0:
            raise ContractError("resource claims require positive replica and instance counts")
        if self.serving_devices != self.serving_replicas * self.devices_per_replica:
            raise ContractError("serving_devices does not match the replica shape")
        if self.allocated_devices < self.serving_devices:
            raise ContractError("allocated_devices cannot be smaller than serving_devices")
        if self.idle_devices != self.allocated_devices - self.serving_devices:
            raise ContractError("idle_devices does not match allocated minus serving devices")

    def to_dict(self) -> dict[str, Any]:
        return {
            "serving_replicas": self.serving_replicas,
            "required_instances": self.required_instances,
            "serving_devices": self.serving_devices,
            "allocated_devices": self.allocated_devices,
            "idle_devices": self.idle_devices,
            "devices_per_replica": self.devices_per_replica,
            "replicas_per_instance": self.replicas_per_instance,
            "available_instances": self.available_instances,
            "available_replicas": self.available_replicas,
            "fits_available_inventory": self.fits_available_inventory,
        }


@dataclass(frozen=True, slots=True)
class SolverCandidate:
    rank: int
    candidate_id: str
    provider: ProviderInstanceSpec
    runtime_option: RuntimeOption
    status: CandidateStatus
    confidence: str
    context_tokens: int
    recipe_variant_fingerprint: str | None
    sequences_per_replica: int
    replicas_per_instance: int
    sequences_per_instance: int
    max_replicas: int | None
    max_sequences_at_context: int | None
    resource_claim: ResourceClaim | None
    hourly_cost: float | None
    cost_currency: str | None
    rejection_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    contract: CapacityContract | None
    single_group_audit: RecipeAudit | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "provider": self.provider.to_dict(),
            "runtime_option": self.runtime_option.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "context_tokens": self.context_tokens,
            "recipe_variant_fingerprint": self.recipe_variant_fingerprint,
            "sequences_per_replica": self.sequences_per_replica,
            "replicas_per_instance": self.replicas_per_instance,
            "sequences_per_instance": self.sequences_per_instance,
            "max_replicas": self.max_replicas,
            "max_sequences_at_context": self.max_sequences_at_context,
            "resource_claim": None if self.resource_claim is None else self.resource_claim.to_dict(),
            "hourly_cost": self.hourly_cost,
            "cost_currency": self.cost_currency,
            "rejection_reasons": list(self.rejection_reasons),
            "warnings": list(self.warnings),
            "contract": None if self.contract is None else self.contract.to_dict(),
            "single_group_audit": (None if self.single_group_audit is None else self.single_group_audit.to_dict()),
        }


@dataclass(frozen=True, slots=True)
class ExplorationResult:
    schema_version: str
    model: ModelSpec
    model_manifest: ModelManifest | None
    context_tokens: int
    candidates: tuple[SolverCandidate, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "capacity-exploration-1.0":
            raise ContractError("unsupported capacity exploration schema_version")
        if self.model_manifest is not None and self.model_manifest.model != self.model:
            raise ContractError("model_manifest does not match model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "model_manifest": None if self.model_manifest is None else self.model_manifest.to_dict(),
            "context_tokens": self.context_tokens,
            "candidates": [item.to_dict() for item in self.candidates],
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    schema_version: str
    model: ModelSpec
    model_manifest: ModelManifest | None
    load: LoadRequirement
    objective: PlanningObjective
    candidates: tuple[SolverCandidate, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "capacity-plan-1.0":
            raise ContractError("unsupported capacity plan schema_version")
        if self.model_manifest is not None and self.model_manifest.model != self.model:
            raise ContractError("model_manifest does not match model")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "model_manifest": None if self.model_manifest is None else self.model_manifest.to_dict(),
            "load": self.load.to_dict(),
            "objective": self.objective.value,
            "candidates": [item.to_dict() for item in self.candidates],
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True, slots=True)
class _Evaluation:
    provider: ProviderInstanceSpec
    runtime_option: RuntimeOption
    contract: CapacityContract | None
    recipe: ServingRecipe | None
    single_group_audit: RecipeAudit | None
    sequences_per_replica: int
    replicas_per_instance: int
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def candidate_id(self) -> str:
        return f"{self.provider.inventory_key}/{self.runtime_option.runtime_id}"


def explore(
    model: ModelSpec | ModelManifest,
    providers: ProviderInventory,
    runtimes: RuntimeInventory,
    *,
    context_tokens: int,
) -> ExplorationResult:
    """Compare context and sequence capacity without predicting traffic."""

    model_spec, manifest = _model_input(model)
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
        raise ContractError("context_tokens must be a positive integer")
    candidates = [
        _exploration_candidate(
            _evaluate(model_spec, provider, runtime, context_tokens, manifest),
            context_tokens,
        )
        for provider in providers.items
        for runtime in runtimes.items
    ]
    return ExplorationResult(
        "capacity-exploration-1.0",
        model_spec,
        manifest,
        context_tokens,
        _rank(candidates, None),
        (
            "Results check model memory, context, runtime, and routing limits; "
            "they do not predict throughput or latency.",
            "Runtime and accelerator support comes from caller-supplied runtime options.",
            "Each candidate uses one provider and one runtime option; cross-provider placement is not performed.",
        ),
    )


def plan(
    model: ModelSpec | ModelManifest,
    providers: ProviderInventory,
    runtimes: RuntimeInventory,
    load: LoadRequirement,
    *,
    objective: PlanningObjective = PlanningObjective.LOWEST_COST,
    measurements: MeasurementInventory | None = None,
) -> CapacityPlan:
    """Size one model for each supplied provider and runtime option."""

    model_spec, manifest = _model_input(model)
    try:
        chosen_objective = PlanningObjective(objective)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"unsupported planning objective {objective!r}") from exc
    if load.measured_profile is not None and measurements is not None:
        raise ContractError("supply measurements on the load or in a measurement inventory, not both")
    currencies = {
        item.price_currency
        for item in providers.items
        if item.price_per_instance_hour is not None and item.price_currency is not None
    }
    if chosen_objective == PlanningObjective.LOWEST_COST and len(currencies) > 1:
        raise ContractError("lowest-cost planning cannot compare provider prices in different currencies")

    candidates = [
        _plan_candidate(
            _evaluate(model_spec, provider, runtime, load.context_tokens, manifest),
            load,
            measurements,
        )
        for provider in providers.items
        for runtime in runtimes.items
    ]
    return CapacityPlan(
        "capacity-plan-1.0",
        model_spec,
        manifest,
        load,
        chosen_objective,
        _rank(candidates, chosen_objective),
        (
            "Memory and context checks are analytical; runtime initialization remains the fit gate.",
            "RPS, TPS, TTFT, and TPOT sizing requires a measured profile for the exact recipe and operating point.",
            "Prices and availability are caller-supplied snapshots, not reservations or live quotes.",
            "Each candidate is homogeneous; the planner does not split one model across providers.",
        ),
    )


def _evaluate(
    model: ModelSpec,
    provider: ProviderInstanceSpec,
    runtime_option: RuntimeOption,
    context_tokens: int,
    manifest: ModelManifest | None,
) -> _Evaluation:
    runtime = runtime_option.runtime
    tp = runtime.tensor_parallel_size
    warnings: list[str] = []
    if runtime_option.supported_accelerator_ids:
        if provider.accelerator_id not in runtime_option.supported_accelerator_ids:
            return _rejected(
                provider,
                runtime_option,
                ("runtime option does not support the provider accelerator",),
            )
    else:
        warnings.append("runtime compatibility with this accelerator is caller-assumed, not verified")
    if tp > provider.devices_per_instance:
        return _rejected(
            provider,
            runtime_option,
            ("runtime tensor parallelism exceeds devices_per_instance",),
            warnings,
        )

    replicas_per_instance = provider.devices_per_instance // tp
    unused_devices = provider.devices_per_instance % tp
    if unused_devices:
        warnings.append(f"{unused_devices} device(s) per instance cannot be assigned to a TP={tp} replica")
    if (
        manifest is not None
        and model.weight_dtype in {"int4", "int8", "fp8", "quantized"}
        and runtime.weight_bytes_per_device_override is None
        and (model.explicit_weight_bytes is None or tp > 1)
    ):
        return _rejected(
            provider,
            runtime_option,
            ("quantized model requires measured per-device resident weight bytes for this parallel layout",),
            warnings,
            replicas_per_instance,
        )
    try:
        contract = capacity_for(model, provider.hardware_slice(tp), runtime)
    except ContractError as exc:
        return _rejected(provider, runtime_option, (str(exc),), warnings, replicas_per_instance)
    warnings.extend(contract.warnings)
    if not contract.fits:
        return _rejected(
            provider,
            runtime_option,
            ("model and runtime do not fit the provider instance slice",),
            warnings,
            replicas_per_instance,
            contract,
        )

    recipe = _recipe(model, provider, runtime_option, contract, manifest)
    single_group_load = LoadRequirement("load-requirement-1.0", context_tokens)
    audit = audit_recipe(recipe, single_group_load)
    configuration_issues = tuple(audit.issues)
    if audit.status == AuditStatus.INVALID or configuration_issues:
        return _rejected(
            provider,
            runtime_option,
            configuration_issues,
            (*warnings, *audit.warnings),
            replicas_per_instance,
            contract,
            recipe,
            audit,
        )
    sequences = audit.effective_sequences_per_group
    if sequences <= 0:
        return _rejected(
            provider,
            runtime_option,
            ("requested context has no sequence capacity for this candidate",),
            (*warnings, *audit.warnings),
            replicas_per_instance,
            contract,
            recipe,
            audit,
        )
    return _Evaluation(
        provider,
        runtime_option,
        contract,
        recipe,
        audit,
        sequences,
        replicas_per_instance,
        tuple((*warnings, *audit.warnings)),
        (),
    )


def _model_input(model: ModelSpec | ModelManifest) -> tuple[ModelSpec, ModelManifest | None]:
    if isinstance(model, ModelManifest):
        return model.model, model
    if isinstance(model, ModelSpec):
        return model, None
    raise ContractError("model must be a ModelSpec or ModelManifest")


def _recipe(
    model: ModelSpec,
    provider: ProviderInstanceSpec,
    runtime_option: RuntimeOption,
    contract: CapacityContract,
    manifest: ModelManifest | None,
) -> ServingRecipe:
    runtime = runtime_option.runtime
    tp = runtime.tensor_parallel_size
    return ServingRecipe(
        schema_version="serving-recipe-1.0",
        recipe_id=f"{provider.inventory_key}/{runtime_option.runtime_id}",
        model=model,
        host=contract.hardware,
        runtime=runtime,
        topology=ParallelTopology(
            tensor_parallel_size=tp,
            data_parallel_size=1,
            expert_parallel_size=runtime_option.expert_parallel_size,
            physical_device_count=tp,
            independent_kv_ranks=1,
        ),
        llmd=runtime_option.routing,
        configured_groups=1,
        evidence=(
            *(() if manifest is None else manifest.evidence),
            EvidenceRecord(
                evidence_id=f"provider-{provider.inventory_key}",
                kind=EvidenceKind.REPORTED,
                source=provider.source,
                scope="caller-supplied provider instance inventory",
                collected_at=provider.collected_at,
            ),
            EvidenceRecord(
                evidence_id=f"runtime-{runtime_option.runtime_id}",
                kind=EvidenceKind.REPORTED,
                source=runtime_option.source,
                scope="caller-supplied runtime option",
            ),
        ),
    )


def _exploration_candidate(evaluation: _Evaluation, context_tokens: int) -> SolverCandidate:
    provider = evaluation.provider
    max_replicas = _max_replicas(evaluation)
    return SolverCandidate(
        rank=0,
        candidate_id=evaluation.candidate_id,
        provider=provider,
        runtime_option=evaluation.runtime_option,
        status=CandidateStatus.REJECTED if evaluation.rejection_reasons else CandidateStatus.FITS,
        confidence="analytical-memory",
        context_tokens=context_tokens,
        recipe_variant_fingerprint=None if evaluation.recipe is None else evaluation.recipe.variant_fingerprint,
        sequences_per_replica=evaluation.sequences_per_replica,
        replicas_per_instance=evaluation.replicas_per_instance,
        sequences_per_instance=evaluation.sequences_per_replica * evaluation.replicas_per_instance,
        max_replicas=max_replicas,
        max_sequences_at_context=(None if max_replicas is None else max_replicas * evaluation.sequences_per_replica),
        resource_claim=None,
        hourly_cost=None,
        cost_currency=provider.price_currency,
        rejection_reasons=evaluation.rejection_reasons,
        warnings=evaluation.warnings,
        contract=evaluation.contract,
        single_group_audit=evaluation.single_group_audit,
    )


def _plan_candidate(
    evaluation: _Evaluation,
    load: LoadRequirement,
    measurements: MeasurementInventory | None,
) -> SolverCandidate:
    provider = evaluation.provider
    if evaluation.rejection_reasons or evaluation.recipe is None:
        return _exploration_candidate(evaluation, load.context_tokens)

    profile = (
        load.measured_profile
        if load.measured_profile is not None
        and load.measured_profile.recipe_variant_fingerprint == evaluation.recipe.variant_fingerprint
        else None
    )
    if measurements is not None:
        profile = measurements.for_recipe(
            evaluation.recipe.variant_fingerprint,
            load.context_tokens,
            load.input_tokens_per_request,
            load.output_tokens_per_request,
        )
    candidate_load = replace(load, measured_profile=profile)
    audit = audit_recipe(evaluation.recipe, candidate_load)
    scale_issue = "configured_groups=1 is below required_groups="
    blocking_issues = tuple(issue for issue in audit.issues if not issue.startswith(scale_issue))
    warnings = list(evaluation.warnings)
    if load.measured_profile is not None and profile is None:
        warnings.append("the load's measured profile belongs to another candidate and was not reused")
    if profile is None and _needs_measurement(load):
        warnings.append("no measured profile matches this candidate and operating point")

    claim: ResourceClaim | None = None
    hourly_cost: float | None = None
    rejection_reasons: tuple[str, ...] = ()
    if audit.status == AuditStatus.INVALID or blocking_issues:
        status = CandidateStatus.REJECTED
        rejection_reasons = blocking_issues or audit.issues
    elif audit.required_groups is None:
        status = CandidateStatus.INCOMPLETE
    else:
        claim = _resource_claim(evaluation, audit.required_groups)
        status = (
            CandidateStatus.INSUFFICIENT_INVENTORY
            if claim.fits_available_inventory is False
            else CandidateStatus.FEASIBLE
        )
        if provider.price_per_instance_hour is not None:
            hourly_cost = claim.required_instances * provider.price_per_instance_hour
        if provider.max_instances is None:
            warnings.append("provider availability is unknown")
        if provider.price_per_instance_hour is None:
            warnings.append("provider price is unknown")

    max_replicas = _max_replicas(evaluation)
    return SolverCandidate(
        rank=0,
        candidate_id=evaluation.candidate_id,
        provider=provider,
        runtime_option=evaluation.runtime_option,
        status=status,
        confidence=("analytical-memory+measured-performance" if profile is not None else "analytical-memory"),
        context_tokens=load.context_tokens,
        recipe_variant_fingerprint=evaluation.recipe.variant_fingerprint,
        sequences_per_replica=evaluation.sequences_per_replica,
        replicas_per_instance=evaluation.replicas_per_instance,
        sequences_per_instance=evaluation.sequences_per_replica * evaluation.replicas_per_instance,
        max_replicas=max_replicas,
        max_sequences_at_context=(None if max_replicas is None else max_replicas * evaluation.sequences_per_replica),
        resource_claim=claim,
        hourly_cost=hourly_cost,
        cost_currency=provider.price_currency,
        rejection_reasons=rejection_reasons,
        warnings=tuple(dict.fromkeys((*warnings, *audit.warnings))),
        contract=evaluation.contract,
        single_group_audit=audit,
    )


def _resource_claim(evaluation: _Evaluation, replicas: int) -> ResourceClaim:
    provider = evaluation.provider
    runtime = evaluation.runtime_option.runtime
    instances = ceil_div(replicas, evaluation.replicas_per_instance)
    serving_devices = replicas * runtime.tensor_parallel_size
    allocated_devices = instances * provider.devices_per_instance
    available_replicas = (
        None if provider.max_instances is None else provider.max_instances * evaluation.replicas_per_instance
    )
    return ResourceClaim(
        serving_replicas=replicas,
        required_instances=instances,
        serving_devices=serving_devices,
        allocated_devices=allocated_devices,
        idle_devices=allocated_devices - serving_devices,
        devices_per_replica=runtime.tensor_parallel_size,
        replicas_per_instance=evaluation.replicas_per_instance,
        available_instances=provider.max_instances,
        available_replicas=available_replicas,
        fits_available_inventory=(None if provider.max_instances is None else instances <= provider.max_instances),
    )


def _max_replicas(evaluation: _Evaluation) -> int | None:
    if evaluation.provider.max_instances is None:
        return None
    return evaluation.provider.max_instances * evaluation.replicas_per_instance


def _needs_measurement(load: LoadRequirement) -> bool:
    return any(
        (
            load.request_rate_per_second,
            load.effective_prefill_tokens_per_second,
            load.effective_decode_tokens_per_second,
            load.target_ttft_ms,
            load.target_tpot_ms,
        )
    )


def _rejected(
    provider: ProviderInstanceSpec,
    runtime_option: RuntimeOption,
    reasons: tuple[str, ...],
    warnings: list[str] | tuple[str, ...] = (),
    replicas_per_instance: int = 0,
    contract: CapacityContract | None = None,
    recipe: ServingRecipe | None = None,
    audit: RecipeAudit | None = None,
) -> _Evaluation:
    return _Evaluation(
        provider,
        runtime_option,
        contract,
        recipe,
        audit,
        0,
        replicas_per_instance,
        tuple(warnings),
        reasons,
    )


def _rank(
    candidates: list[SolverCandidate],
    objective: PlanningObjective | None,
) -> tuple[SolverCandidate, ...]:
    status_order = {
        CandidateStatus.FEASIBLE: 0,
        CandidateStatus.FITS: 0,
        CandidateStatus.INSUFFICIENT_INVENTORY: 1,
        CandidateStatus.INCOMPLETE: 2,
        CandidateStatus.REJECTED: 3,
    }

    def key(candidate: SolverCandidate) -> tuple[object, ...]:
        order: tuple[object, ...]
        if objective == PlanningObjective.LOWEST_COST:
            order = (
                candidate.hourly_cost is None,
                float("inf") if candidate.hourly_cost is None else candidate.hourly_cost,
            )
        elif objective == PlanningObjective.FEWEST_ALLOCATED_DEVICES:
            devices = None if candidate.resource_claim is None else candidate.resource_claim.allocated_devices
            order = (devices is None, float("inf") if devices is None else devices)
        else:
            order = (-candidate.sequences_per_instance,)
        return (status_order[candidate.status], *order, candidate.candidate_id)

    ordered = sorted(candidates, key=key)
    return tuple(replace(item, rank=index) for index, item in enumerate(ordered, start=1))
