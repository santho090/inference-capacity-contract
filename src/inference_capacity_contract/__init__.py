"""Portable, deterministic capacity contracts for LLM inference."""

from .adapters import to_llmd_planner_payload, to_scaling_policy_input
from .calculator import capacity_for, what_fits
from .models import (
    CapacityContract,
    ConcurrencyPoint,
    ContractError,
    EvidenceKind,
    EvidenceRecord,
    HardwareInventory,
    HardwareSpec,
    ModelSpec,
    RuntimeVariant,
    ValidationLevel,
    WorkloadProfile,
)
from .recipe import (
    RECIPE_VARIANT_FINGERPRINT_VERSION,
    AuditStatus,
    LLMDRoutingSpec,
    LoadRequirement,
    MeasuredGroupProfile,
    ParallelTopology,
    RecipeAudit,
    ServingRecipe,
    audit_recipe,
)
from .scaling import ScalingRecommendation, recommend_scale

__all__ = [
    "CapacityContract",
    "ConcurrencyPoint",
    "ContractError",
    "EvidenceKind",
    "EvidenceRecord",
    "HardwareInventory",
    "HardwareSpec",
    "ModelSpec",
    "RuntimeVariant",
    "ValidationLevel",
    "capacity_for",
    "what_fits",
    "to_llmd_planner_payload",
    "to_scaling_policy_input",
    "ScalingRecommendation",
    "WorkloadProfile",
    "recommend_scale",
    "AuditStatus",
    "LLMDRoutingSpec",
    "LoadRequirement",
    "MeasuredGroupProfile",
    "ParallelTopology",
    "RecipeAudit",
    "RECIPE_VARIANT_FINGERPRINT_VERSION",
    "ServingRecipe",
    "audit_recipe",
]
