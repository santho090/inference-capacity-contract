"""Adapter for exact values exported by a vLLM initialization run."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .importers import VLLMInitializationProfile, import_vllm_initialization
from .models import ContractError

SNAPSHOT_SCHEMA_VERSION = "vllm-runtime-snapshot-1.0"
_SNAPSHOT_FIELDS = {
    "schema_version",
    "model_revision",
    "hardware_id",
    "runtime_version",
    "tensor_parallel_size",
    "data_parallel_size",
    "expert_parallel_size",
    "physical_device_count",
    "memory_utilization_limit",
    "kv_cache_dtype",
    "block_size_tokens",
    "max_num_seqs",
    "context_tokens",
    "kv_cache_size_tokens",
    "kv_cache_max_concurrency",
    "workers",
}
_WORKER_FIELDS = {
    "worker_rank",
    "device_memory_bytes",
    "requested_memory_bytes",
    "model_memory_usage_bytes",
    "peak_activation_memory_bytes",
    "available_kv_cache_memory_bytes",
}


def import_vllm_runtime_snapshot(
    data: Mapping[str, Any],
    *,
    source: str,
) -> VLLMInitializationProfile:
    """Turn exact vLLM worker and scheduler values into an ICC initialization profile."""

    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ContractError(f"unsupported vLLM runtime snapshot schema_version; expected {SNAPSHOT_SCHEMA_VERSION!r}")
    _reject_unknown(data, _SNAPSHOT_FIELDS, "vLLM runtime snapshot")
    workers = _workers(data.get("workers"))
    engine_ranks = _positive_int(data, "tensor_parallel_size") * _positive_int(data, "data_parallel_size")
    if len(workers) != engine_ranks:
        raise ContractError("vLLM runtime snapshot must contain one worker entry for every TP x DP rank")
    if {worker["worker_rank"] for worker in workers} != set(range(engine_ranks)):
        raise ContractError("vLLM runtime snapshot worker ranks must cover 0 through TP x DP minus one")
    utilization = _positive_number(data, "memory_utilization_limit")
    if utilization > 1:
        raise ContractError("memory_utilization_limit must be in (0, 1]")
    device_sizes = {worker["device_memory_bytes"] for worker in workers}
    if len(device_sizes) != 1:
        raise ContractError("vLLM workers have different device memory sizes; one hardware profile is unsafe")
    if any(
        worker["requested_memory_bytes"] != math.ceil(worker["device_memory_bytes"] * utilization) for worker in workers
    ):
        raise ContractError("vLLM requested memory does not match device memory x utilization")
    context_tokens = _positive_int(data, "context_tokens")
    max_num_seqs = _positive_int(data, "max_num_seqs")
    kv_cache_size_tokens = _positive_int(data, "kv_cache_size_tokens")
    max_concurrency = _positive_number(data, "kv_cache_max_concurrency")
    if kv_cache_size_tokens != int(max_concurrency * context_tokens):
        raise ContractError("vLLM KV cache token size does not match max concurrency x context")
    max_sequences = min(math.floor(max_concurrency), max_num_seqs)
    if max_sequences <= 0:
        raise ContractError("vLLM runtime snapshot cannot hold one sequence at the requested context")

    limiting = min(workers, key=lambda worker: (worker["available_kv_cache_memory_bytes"], worker["worker_rank"]))
    runtime_overhead = (
        limiting["requested_memory_bytes"]
        - limiting["model_memory_usage_bytes"]
        - limiting["peak_activation_memory_bytes"]
        - limiting["available_kv_cache_memory_bytes"]
    )
    if runtime_overhead < 0:
        raise ContractError("vLLM worker memory components exceed its requested memory budget")

    normalized = {
        "model_revision": data.get("model_revision"),
        "hardware_id": data.get("hardware_id"),
        "runtime_engine": "vllm",
        "runtime_version": data.get("runtime_version"),
        "tensor_parallel_size": data.get("tensor_parallel_size"),
        "data_parallel_size": data.get("data_parallel_size"),
        "expert_parallel_size": data.get("expert_parallel_size"),
        "physical_device_count": data.get("physical_device_count"),
        "memory_utilization_limit": data.get("memory_utilization_limit"),
        "kv_cache_dtype": data.get("kv_cache_dtype"),
        "block_size_tokens": data.get("block_size_tokens"),
        "max_num_seqs": max_num_seqs,
        "weight_bytes_per_device": limiting["model_memory_usage_bytes"],
        "runtime_overhead_bytes_per_device": runtime_overhead,
        "activation_reserve_bytes_per_device": limiting["peak_activation_memory_bytes"],
        "kv_capacity_memory_bytes_per_device": limiting["available_kv_cache_memory_bytes"],
        "kv_capacity_envelope": [{"context_tokens": context_tokens, "max_sequences": max_sequences}],
    }
    profile = import_vllm_initialization(normalized, source=source)
    snapshot_document = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        **{key: normalized[key] for key in _IDENTITY_FIELDS},
        "context_tokens": context_tokens,
        "kv_cache_size_tokens": kv_cache_size_tokens,
        "kv_cache_max_concurrency": max_concurrency,
        "workers": workers,
    }
    digest_input = json.dumps(snapshot_document, separators=(",", ":"), sort_keys=True).encode()
    evidence = replace(
        profile.evidence,
        metrics={
            **profile.evidence.metrics,
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_digest": f"sha256:{hashlib.sha256(digest_input).hexdigest()}",
            "worker_count": len(workers),
            "limiting_worker_rank": limiting["worker_rank"],
            "kv_cache_size_tokens": kv_cache_size_tokens,
            "kv_cache_max_concurrency": max_concurrency,
        },
    )
    return replace(profile, evidence=evidence)


_IDENTITY_FIELDS = (
    "model_revision",
    "hardware_id",
    "runtime_version",
    "tensor_parallel_size",
    "data_parallel_size",
    "expert_parallel_size",
    "physical_device_count",
    "memory_utilization_limit",
    "kv_cache_dtype",
    "block_size_tokens",
    "max_num_seqs",
)


def _workers(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise ContractError("vLLM runtime snapshot workers must be a non-empty array")
    workers: list[dict[str, int]] = []
    ranks: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise ContractError("vLLM runtime snapshot workers must contain objects")
        _reject_unknown(item, _WORKER_FIELDS, "vLLM worker snapshot")
        worker = {
            "worker_rank": _non_negative_int(item, "worker_rank"),
            "device_memory_bytes": _positive_int(item, "device_memory_bytes"),
            "requested_memory_bytes": _positive_int(item, "requested_memory_bytes"),
            "model_memory_usage_bytes": _positive_int(item, "model_memory_usage_bytes"),
            "peak_activation_memory_bytes": _non_negative_int(item, "peak_activation_memory_bytes"),
            "available_kv_cache_memory_bytes": _positive_int(item, "available_kv_cache_memory_bytes"),
        }
        if worker["worker_rank"] in ranks:
            raise ContractError("vLLM runtime snapshot worker ranks must be unique")
        ranks.add(worker["worker_rank"])
        workers.append(worker)
    return sorted(workers, key=lambda worker: worker["worker_rank"])


def _positive_int(data: Mapping[str, Any], name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _non_negative_int(data: Mapping[str, Any], name: str) -> int:
    value = data.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _positive_number(data: Mapping[str, Any], name: str) -> float:
    value = data.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ContractError(f"{name} must be a finite positive number")
    return float(value)


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], scope: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ContractError(f"{scope} contains unknown field(s): {', '.join(unknown)}")
