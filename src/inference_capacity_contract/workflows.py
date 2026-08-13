"""End-to-end workflows built from the model resolver and provider planner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .importers import (
    IncompleteModelError,
    JSONFetcher,
    ModelResolutionDraft,
    RangeFetcher,
    resolve_huggingface_manifest,
)
from .inventory import MeasurementInventory, ProviderInventory, RuntimeInventory
from .models import ContractError
from .planner import CapacityPlan, PlanningObjective, plan
from .recipe import LoadRequirement


class ModelPlanningStatus(StrEnum):
    NEEDS_MODEL_INPUTS = "needs-model-inputs"
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class ModelPlanningResult:
    """One-call result that is either actionable missing input or a capacity plan."""

    schema_version: str
    status: ModelPlanningStatus
    model_resolution: ModelResolutionDraft | None
    capacity_plan: CapacityPlan | None

    def __post_init__(self) -> None:
        if self.schema_version != "model-planning-result-1.0":
            raise ContractError("unsupported model planning result schema_version")
        try:
            status = ModelPlanningStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"unsupported model planning status {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        if self.model_resolution is not None and not isinstance(self.model_resolution, ModelResolutionDraft):
            raise ContractError("model_resolution must be a ModelResolutionDraft")
        if self.capacity_plan is not None and not isinstance(self.capacity_plan, CapacityPlan):
            raise ContractError("capacity_plan must be a CapacityPlan")
        if status == ModelPlanningStatus.NEEDS_MODEL_INPUTS:
            if self.model_resolution is None or self.model_resolution.ready or self.capacity_plan is not None:
                raise ContractError("needs-model-inputs requires one unresolved model draft and no capacity plan")
        elif self.capacity_plan is None or self.model_resolution is not None:
            raise ContractError("planned requires one capacity plan and no model draft")
        elif self.capacity_plan.model_manifest is None:
            raise ContractError("planned model workflow requires a replayable model manifest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "model_resolution": None if self.model_resolution is None else self.model_resolution.to_dict(),
            "capacity_plan": None if self.capacity_plan is None else self.capacity_plan.to_dict(),
        }


def plan_huggingface_model(
    repo_id: str,
    providers: ProviderInventory,
    runtimes: RuntimeInventory,
    load: LoadRequirement,
    *,
    revision: str = "main",
    objective: PlanningObjective = PlanningObjective.LOWEST_COST,
    measurements: MeasurementInventory | None = None,
    fetch_json: JSONFetcher | None = None,
    fetch_range: RangeFetcher | None = None,
    cache_dir: Path | None = None,
    kv_bytes_per_token_per_device_override: int | None = None,
    parameter_count_override: int | None = None,
    resident_weight_bytes_override: int | None = None,
    weight_dtype_override: str | None = None,
    inspect_safetensors_headers: bool = True,
) -> ModelPlanningResult:
    """Resolve one public model and plan it across caller-supplied candidates."""

    try:
        manifest = resolve_huggingface_manifest(
            repo_id,
            revision,
            fetch_json=fetch_json,
            fetch_range=fetch_range,
            cache_dir=cache_dir,
            kv_bytes_per_token_per_device_override=kv_bytes_per_token_per_device_override,
            parameter_count_override=parameter_count_override,
            resident_weight_bytes_override=resident_weight_bytes_override,
            weight_dtype_override=weight_dtype_override,
            inspect_safetensors_headers=inspect_safetensors_headers,
        )
    except IncompleteModelError as exc:
        return ModelPlanningResult(
            "model-planning-result-1.0",
            ModelPlanningStatus.NEEDS_MODEL_INPUTS,
            exc.draft,
            None,
        )
    return ModelPlanningResult(
        "model-planning-result-1.0",
        ModelPlanningStatus.PLANNED,
        None,
        plan(
            manifest,
            providers,
            runtimes,
            load,
            objective=objective,
            measurements=measurements,
        ),
    )
