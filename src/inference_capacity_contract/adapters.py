"""Small, non-actuating export adapters for upstream schedulers.

The payloads are intentionally neutral. They carry facts and bounds that a
planner or autoscaler can consume; they do not emit deployment YAML, mutate a
cluster, or choose a replica count without measured workload evidence.
"""

from __future__ import annotations

from typing import Any

from .models import CapacityContract


def to_llmd_planner_payload(contract: CapacityContract) -> dict[str, Any]:
    """Export a capacity fact set suitable for an llm-d planner adapter."""

    return {
        "schema_version": "llmd-capacity-input-2.0",
        "producer": "inference-capacity-contract",
        "subject": {
            "model": contract.model.to_dict(),
            "hardware": contract.hardware.to_dict(),
            "runtime": contract.runtime.to_dict(),
        },
        "feasibility": {
            "fits": contract.fits,
            "validation_level": contract.validation_level.value,
            "warnings": list(contract.warnings),
        },
        "memory": dict(contract.memory_bytes_per_device),
        "kv": {
            "bytes_per_token_per_device": contract.kv_bytes_per_token_per_device,
            "block_size_tokens": contract.kv_block_size_tokens,
            "capacity_blocks_per_device": contract.kv_capacity_blocks_per_device,
            "capacity_tokens_per_device": contract.kv_capacity_tokens_per_device,
            "max_context_tokens": contract.max_context_tokens,
            "concurrency_envelope": [point.to_dict() for point in contract.concurrency_envelope],
        },
        "evidence": [record.to_dict() for record in contract.evidence],
    }


def to_scaling_policy_input(contract: CapacityContract) -> dict[str, Any]:
    """Export per-replica bounds for a WVA/KEDA-style policy adapter.

    No desired replica count is invented here. Queue-to-replica conversion
    needs an observed throughput or latency profile, so the payload marks that
    missing input explicitly.
    """

    return {
        "schema_version": "scaling-policy-input-2.0",
        "producer": "inference-capacity-contract",
        "selector": {
            "model_id": contract.model.model_id,
            "revision": contract.model.revision,
            "hardware_id": contract.hardware.hardware_id,
            "engine": contract.runtime.engine,
            "runtime_version": contract.runtime.version,
        },
        "per_replica_capacity": {
            "fits": contract.fits,
            "kv_capacity_tokens_per_device": contract.kv_capacity_tokens_per_device,
            "max_context_tokens": contract.max_context_tokens,
            "concurrency_envelope": [point.to_dict() for point in contract.concurrency_envelope],
        },
        "replica_count": None,
        "requires_observed_workload_profile": True,
        "validation_level": contract.validation_level.value,
        "evidence": [record.to_dict() for record in contract.evidence],
        "warnings": list(contract.warnings),
    }
