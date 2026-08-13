"""Import partial llm-d values without turning unknown facts into guesses."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .arithmetic import block_aligned_sequence_capacity
from .models import ContractError, EvidenceKind, EvidenceRecord, HardwareSpec
from .recipe import ServingRecipe


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _first_container(values: Mapping[str, Any]) -> Mapping[str, Any]:
    modelservice = _mapping(values.get("modelservice"))
    decode = _mapping(modelservice.get("decode"))
    containers = decode.get("containers")
    if isinstance(containers, Sequence) and not isinstance(containers, (str, bytes)) and containers:
        normalized = tuple(_mapping(item) for item in containers)
        for container in normalized:
            name = container.get("name")
            image = container.get("image")
            identity = f"{name or ''} {image or ''}".lower()
            if "vllm" in identity or "sglang" in identity:
                return container
        return normalized[0]
    return {}


def _command(container: Mapping[str, Any]) -> str:
    args = container.get("args")
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        return ""
    return " ".join(item for item in args if isinstance(item, str)).replace("\\\n", " ")


def _flag_value(command: str, *names: str) -> str | None:
    alternatives = "|".join(re.escape(name) for name in names)
    match = re.search(rf"(?:^|\s)(?:{alternatives})(?:=|\s+)([^\s\\]+)", command)
    return None if match is None else match.group(1).strip("'\"")


def _flag_int(command: str, *names: str) -> int | None:
    raw = _flag_value(command, *names)
    if raw is None or not raw.isdigit():
        return None
    return _positive_int(int(raw))


def _flag_float(command: str, *names: str) -> float | None:
    raw = _flag_value(command, *names)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if 0 < value <= 1 else None


def _split_model_reference(value: object) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, None
    reference = value.removeprefix("hf://")
    model_id, separator, revision = reference.rpartition("@")
    if separator and re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return model_id, revision.lower()
    return reference, None


def _model_reference(
    values: Mapping[str, Any], metadata: Mapping[str, Any], command: str
) -> tuple[str | None, str | None]:
    direct_identity: str | None = None
    artifact_name: str | None = None
    artifact_uri: str | None = None
    revisions: list[str] = []
    direct_id, direct_revision = _split_model_reference(metadata.get("model_name"))
    if direct_id is not None:
        direct_identity = direct_id
    if direct_revision is not None:
        revisions.append(direct_revision)
    metadata_revision = metadata.get("model_revision")
    if isinstance(metadata_revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", metadata_revision):
        revisions.append(metadata_revision.lower())
    artifacts = _mapping(_mapping(values.get("modelservice")).get("modelArtifacts"))
    for key in ("name", "uri"):
        model_id, revision = _split_model_reference(artifacts.get(key))
        if model_id is not None:
            if key == "name":
                artifact_name = model_id
            else:
                artifact_uri = model_id
        if revision is not None:
            revisions.append(revision)
    artifact_revision = artifacts.get("revision")
    if isinstance(artifact_revision, str) and re.fullmatch(r"[0-9a-fA-F]{40}", artifact_revision):
        revisions.append(artifact_revision.lower())
    command_revision = _flag_value(command, "--revision")
    if command_revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", command_revision):
        revisions.append(command_revision.lower())
    if len(set(revisions)) > 1:
        raise ContractError("llm-d values contain conflicting immutable model revisions")
    identity = artifact_uri or direct_identity or artifact_name
    return identity, revisions[0] if revisions else None


def _runtime_identity(container: Mapping[str, Any]) -> tuple[str | None, str | None]:
    image = container.get("image")
    if not isinstance(image, str) or not image:
        return None, None
    engine = "vllm" if "vllm" in image.lower() else "sglang" if "sglang" in image.lower() else "other"
    version = image.rsplit(":", 1)[1] if ":" in image.rsplit("/", 1)[-1] else image
    return engine, version


def _deep_merge(base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in changes.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _path_value(document: Mapping[str, Any], path: str) -> object:
    current: object = document
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _fact_is_unresolved(document: Mapping[str, Any], fact: UnresolvedFact) -> bool:
    if _path_value(document, fact.path) is not None:
        return False
    if fact.path == "model.kv_bytes_per_token_per_device_override":
        return _path_value(document, "model.attention_type") not in {"mha", "gqa", "mqa"}
    if fact.path == "runtime.weight_bytes_per_device_override":
        return _path_value(document, "topology.expert_parallel_size") != 1
    return True


_REQUIRED_FACTS = (
    ("model.model_id", "model identity is absent", "identity"),
    ("model.revision", "the deployment reference is not pinned to an immutable artifact revision", "identity"),
    ("model.parameter_count", "model architecture metadata is absent", "memory"),
    ("model.num_layers", "model architecture metadata is absent", "memory"),
    ("model.num_kv_heads", "model architecture metadata is absent", "memory"),
    ("model.head_dim", "model architecture metadata is absent", "memory"),
    ("model.weight_dtype", "artifact dtype is absent", "memory"),
    ("model.attention_type", "attention layout is absent", "memory"),
    (
        "model.kv_bytes_per_token_per_device_override",
        "custom, MLA, and hybrid cache layouts need a measured per-device KV value",
        "memory",
    ),
    ("host.hardware_id", "host inventory metadata is absent", "topology"),
    ("runtime.engine", "runtime image is absent", "identity"),
    ("runtime.version", "runtime image version is absent", "identity"),
    ("runtime.tensor_parallel_size", "tensor parallelism is absent", "topology"),
    ("runtime.kv_cache_dtype", "KV dtype is absent", "memory"),
    (
        "runtime.weight_bytes_per_device_override",
        "expert-parallel resident bytes per device are absent",
        "memory",
    ),
    ("runtime.runtime_overhead_bytes_per_device", "measured runtime reserve is absent", "memory"),
    ("runtime.activation_reserve_bytes_per_device", "measured activation reserve is absent", "memory"),
    ("runtime.max_num_seqs", "runtime concurrency limit is absent", "topology"),
    ("runtime.block_size_tokens", "runtime KV block size is absent", "routing"),
    ("topology.tensor_parallel_size", "tensor parallelism is absent", "topology"),
    ("topology.data_parallel_size", "data parallelism is absent", "topology"),
    ("topology.physical_device_count", "physical device count is absent", "topology"),
    ("llmd.precise_prefix_routing", "KV events do not prove which endpoint picker is selected", "routing"),
)


def _canonical_unresolved(
    document: Mapping[str, Any], supplied: tuple[UnresolvedFact, ...]
) -> tuple[UnresolvedFact, ...]:
    facts = {item.path: item for item in supplied}
    for values in _REQUIRED_FACTS:
        fact = UnresolvedFact(*values)
        facts.setdefault(fact.path, fact)
    return tuple(item for item in facts.values() if _fact_is_unresolved(document, item))


@dataclass(frozen=True, slots=True)
class UnresolvedFact:
    path: str
    reason: str
    required_for: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path or not isinstance(self.reason, str) or not self.reason:
            raise ContractError("unresolved fact path and reason are required")
        if not isinstance(self.required_for, str) or self.required_for not in {
            "identity",
            "memory",
            "topology",
            "routing",
            "traffic",
        }:
            raise ContractError("unresolved fact required_for is unsupported")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason, "required_for": self.required_for}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UnresolvedFact:
        values: list[str] = []
        for name in ("path", "reason", "required_for"):
            value = data.get(name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"unresolved fact {name} must be a non-empty string")
            values.append(value)
        return cls(*values)


@dataclass(frozen=True, slots=True)
class RecipeDraft:
    """A normalized partial recipe plus the facts required to complete it."""

    schema_version: str
    recipe_id: str
    document: Mapping[str, Any]
    unresolved: tuple[UnresolvedFact, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Keep only facts that are unresolved in the current document."""

        if self.schema_version != "recipe-draft-1.0":
            raise ContractError("unsupported recipe draft schema_version")
        if not isinstance(self.recipe_id, str) or not self.recipe_id:
            raise ContractError("recipe draft recipe_id is required")
        if not isinstance(self.document, Mapping):
            raise ContractError("recipe draft document must be an object")
        if not all(isinstance(item, UnresolvedFact) for item in self.unresolved):
            raise ContractError("recipe draft unresolved entries must be UnresolvedFact values")
        if not all(isinstance(item, str) for item in self.warnings):
            raise ContractError("recipe draft warnings must be strings")
        document = dict(self.document)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "unresolved", _canonical_unresolved(document, tuple(self.unresolved)))

    @property
    def unresolved_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.unresolved)

    @property
    def ready(self) -> bool:
        return not self.unresolved_paths

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "recipe_id": self.recipe_id,
            "document": dict(self.document),
            "unresolved": [item.to_dict() for item in self.unresolved],
            "warnings": list(self.warnings),
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecipeDraft:
        if data.get("schema_version") != "recipe-draft-1.0":
            raise ContractError("unsupported recipe draft schema_version")
        recipe_id = data.get("recipe_id")
        document = data.get("document")
        unresolved = data.get("unresolved")
        warnings = data.get("warnings")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ContractError("recipe draft recipe_id is required")
        if not isinstance(document, Mapping):
            raise ContractError("recipe draft document must be an object")
        if not isinstance(unresolved, list) or not all(isinstance(item, Mapping) for item in unresolved):
            raise ContractError("recipe draft unresolved must be an array of objects")
        if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
            raise ContractError("recipe draft warnings must be an array of strings")
        draft = cls(
            "recipe-draft-1.0",
            recipe_id,
            dict(document),
            tuple(UnresolvedFact.from_dict(item) for item in unresolved),
            tuple(warnings),
        )
        if data.get("ready") is not draft.ready:
            raise ContractError("recipe draft ready flag does not match its unresolved facts")
        return draft

    def with_overrides(self, overrides: Mapping[str, Any]) -> RecipeDraft:
        return RecipeDraft(
            schema_version=self.schema_version,
            recipe_id=self.recipe_id,
            document=_deep_merge(self.document, overrides),
            unresolved=self.unresolved,
            warnings=self.warnings,
        )

    def to_serving_recipe(self, overrides: Mapping[str, Any] | None = None) -> ServingRecipe:
        document = self.document if overrides is None else _deep_merge(self.document, overrides)
        missing = tuple(item.path for item in self.unresolved if _fact_is_unresolved(document, item))
        if missing:
            raise ContractError("cannot build a serving recipe; unresolved facts: " + ", ".join(missing))
        return ServingRecipe.from_dict(document)


class StructuralAuditStatus(StrEnum):
    VALID = "valid"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class StructuralRecipeAudit:
    schema_version: str
    status: StructuralAuditStatus
    engine_rank_count: int | None
    unused_device_count: int | None
    configured_max_concurrent_sequences: int | None
    flow_control_full_context_sequences: int | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    unresolved: tuple[UnresolvedFact, ...]
    suggestions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "engine_rank_count": self.engine_rank_count,
            "unused_device_count": self.unused_device_count,
            "configured_max_concurrent_sequences": self.configured_max_concurrent_sequences,
            "flow_control_full_context_sequences": self.flow_control_full_context_sequences,
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "unresolved": [item.to_dict() for item in self.unresolved],
            "suggestions": list(self.suggestions),
        }


def import_llmd_values(
    values: Mapping[str, Any],
    *,
    host: HardwareSpec | None = None,
    recipe_id: str | None = None,
    source: str = "llmd-values://caller-supplied",
) -> RecipeDraft:
    """Normalize the subset of an llm-d values document that ICC understands.

    The caller parses YAML or JSON. This keeps the core dependency-free and
    makes the trust boundary explicit.
    """

    metadata = _mapping(values.get("metadata"))
    epp = _mapping(metadata.get("epp"))
    container = _first_container(values)
    command = _command(container)
    model_id, model_revision = _model_reference(values, metadata, command)
    engine, runtime_version = _runtime_identity(container)
    tp = _positive_int(metadata.get("tensor_parallel_size")) or _flag_int(command, "--tensor-parallel-size")
    dp = _positive_int(metadata.get("data_parallel_size")) or _flag_int(command, "--data-parallel-size")
    max_num_seqs = _positive_int(metadata.get("max_num_seqs")) or _flag_int(command, "--max-num-seqs")
    max_model_len = _positive_int(metadata.get("max_model_len")) or _flag_int(command, "--max-model-len")
    runtime_block = _flag_int(command, "--block-size")
    llmd_block = _positive_int(metadata.get("kv_block_size"))
    utilization = _flag_float(command, "--gpu-memory-utilization", "--gpu_memory_utilization")
    kv_dtype = _flag_value(command, "--kv-cache-dtype")
    expert_parallel = "--enable-expert-parallel" in command
    kv_events = "--kv-events-config" in command
    physical = None if host is None else host.device_count
    ep = tp * dp if expert_parallel and tp is not None and dp is not None else 1 if not expert_parallel else None
    evidence = EvidenceRecord(
        evidence_id="llmd-values-import",
        kind=EvidenceKind.REPORTED,
        source=source,
        scope="caller-supplied llm-d serving values",
    )
    host_document: dict[str, Any] = {}
    if host is not None:
        host_document = host.to_dict()
        if utilization is not None:
            host_document["memory_utilization_limit"] = utilization

    document: dict[str, Any] = {
        "schema_version": "serving-recipe-1.0",
        "recipe_id": recipe_id or (f"{model_id}-imported" if model_id else "imported-llmd-recipe"),
        "model": {
            "model_id": model_id,
            "revision": model_revision,
            "parameter_count": None,
            "num_layers": None,
            "num_kv_heads": None,
            "head_dim": None,
            "weight_dtype": None,
            "max_model_len": max_model_len,
            "explicit_weight_bytes": None,
            "attention_type": None,
            "kv_bytes_per_token_per_device_override": None,
            "architecture": None,
        },
        "host": host_document,
        "runtime": {
            "engine": engine,
            "version": runtime_version,
            "tensor_parallel_size": tp,
            "kv_cache_dtype": kv_dtype,
            "kv_heads_per_device": None,
            "weight_bytes_per_device_override": None,
            "runtime_overhead_bytes_per_device": None,
            "activation_reserve_bytes_per_device": None,
            "max_num_seqs": max_num_seqs,
            "block_size_tokens": runtime_block or llmd_block,
            "supported_vendors": [] if host is None else [host.vendor],
            "notes": [],
        },
        "topology": {
            "tensor_parallel_size": tp,
            "data_parallel_size": dp,
            "expert_parallel_size": ep,
            "physical_device_count": physical,
            "independent_kv_ranks": dp,
        },
        "llmd": {
            "flow_control_token_limit": _positive_int(epp.get("flow_control_token_limit")),
            "max_concurrent_sequences": _positive_int(epp.get("max_concurrent_sequences")),
            "kv_block_size_tokens": llmd_block or runtime_block,
            "output_ratio": _number(epp.get("output_ratio")),
            "max_estimated_output_tokens": _positive_int(epp.get("max_estimated_output_tokens")),
            "precise_prefix_routing": None if kv_events else False,
        },
        "configured_groups": 1,
        "evidence": [evidence.to_dict()],
    }

    unresolved = tuple(UnresolvedFact(*item) for item in _REQUIRED_FACTS)
    warnings: list[str] = []
    if utilization is not None and host is not None and utilization != host.memory_utilization_limit:
        warnings.append("runtime GPU memory utilization overrides the supplied host default")
    if host is None:
        warnings.append("host inventory facts were not supplied")
    if kv_events:
        warnings.append("KV events are enabled, but the routing policy must be declared separately")
    return RecipeDraft("recipe-draft-1.0", str(document["recipe_id"]), document, unresolved, tuple(warnings))


def audit_recipe_draft(draft: RecipeDraft) -> StructuralRecipeAudit:
    """Audit topology and routing facts that do not depend on model memory."""

    document = draft.document
    topology = _mapping(document.get("topology"))
    runtime = _mapping(document.get("runtime"))
    model = _mapping(document.get("model"))
    llmd = _mapping(document.get("llmd"))
    tp = _positive_int(topology.get("tensor_parallel_size"))
    dp = _positive_int(topology.get("data_parallel_size"))
    physical = _positive_int(topology.get("physical_device_count"))
    kv_ranks = _positive_int(topology.get("independent_kv_ranks"))
    max_num_seqs = _positive_int(runtime.get("max_num_seqs"))
    runtime_block = _positive_int(runtime.get("block_size_tokens"))
    llmd_block = _positive_int(llmd.get("kv_block_size_tokens"))
    flow = _positive_int(llmd.get("flow_control_token_limit"))
    max_model_len = _positive_int(model.get("max_model_len"))
    engine_ranks = tp * dp if tp is not None and dp is not None else None
    unused = physical - engine_ranks if physical is not None and engine_ranks is not None else None
    configured_concurrency = max_num_seqs * kv_ranks if max_num_seqs is not None and kv_ranks is not None else None
    full_context = None
    if flow is not None and max_model_len is not None:
        if llmd_block is None:
            full_context = flow // max_model_len
        else:
            full_context = block_aligned_sequence_capacity(flow, llmd_block, max_model_len)
    issues: list[str] = []
    warnings = list(draft.warnings)
    suggestions: list[str] = []
    if unused is not None and unused < 0:
        issues.append("TP x DP engine ranks exceeds the physical device count")
    elif unused:
        warnings.append(f"{unused} physical devices are not assigned to TP x DP engine ranks")
        suggestions.append("verify the host shape or increase the declared engine-rank topology")
    if runtime_block is not None and llmd_block is not None and runtime_block != llmd_block:
        issues.append("llm-d and runtime KV block sizes do not match")
    if flow is not None and llmd_block is not None and flow % llmd_block:
        warnings.append("flow-control token limit is not divisible by the declared KV block size")
        suggestions.append("compare the flow limit with the runtime-reported block-aligned KV capacity")
    if full_context == 0:
        warnings.append("flow-control budget is smaller than one configured maximum-context sequence")
    unresolved = tuple(item for item in draft.unresolved if _fact_is_unresolved(document, item))
    status = (
        StructuralAuditStatus.INVALID
        if issues
        else StructuralAuditStatus.INCOMPLETE
        if unresolved
        else StructuralAuditStatus.VALID
    )
    return StructuralRecipeAudit(
        "structural-recipe-audit-1.0",
        status,
        engine_ranks,
        unused,
        configured_concurrency,
        full_context,
        tuple(issues),
        tuple(warnings),
        unresolved,
        tuple(suggestions),
    )
