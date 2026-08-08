"""Portable, deterministic capacity contracts for LLM inference."""

from .calculator import capacity_for, what_fits
from .adapters import to_llmd_planner_payload, to_scaling_policy_input
from .scaling import ScalingRecommendation, recommend_scale
from .models import (
    CapacityContract,
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

__all__ = [
    "CapacityContract",
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
]
