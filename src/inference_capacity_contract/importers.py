"""Import immutable model manifests and measured vLLM evidence."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isclose
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from .calculator import capacity_for
from .drafts import RecipeDraft
from .models import DTYPE_BITS, ContractError, EvidenceKind, EvidenceRecord, ModelSpec
from .recipe import (
    RECIPE_VARIANT_FINGERPRINT_VERSION,
    MeasuredGroupProfile,
    ServingRecipe,
)


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _positive_number(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive number")
    return float(value)


def _non_negative_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _required_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _config_int(config: Mapping[str, Any], *names: str, source: str = "config") -> tuple[int, str]:
    for name in names:
        if name in config:
            return _positive_int(name, config[name]), f"{source}.{name}"
    raise ContractError(f"model config is missing {' or '.join(names)}")


def _text_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    nested = config.get("text_config")
    if isinstance(nested, Mapping):
        return nested, "config.text_config"
    return config, "config"


def _quantization_config(config: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    text, text_source = _text_config(config)
    for candidate, source in (
        (config.get("quantization_config"), "config.quantization_config"),
        (text.get("quantization_config"), f"{text_source}.quantization_config"),
    ):
        if isinstance(candidate, Mapping):
            return candidate, source
    return None, None


def _quantization_bits(quantization: Mapping[str, Any], source: str) -> tuple[int | None, str]:
    ignored_modules = quantization.get("ignore")
    if isinstance(ignored_modules, list) and ignored_modules:
        return None, f"{source}.ignore"
    if quantization.get("bits") is not None:
        return _positive_int("quantization_config.bits", quantization.get("bits")), f"{source}.bits"
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping):
        return None, source
    bit_sources: list[tuple[int, str]] = []
    for name, group in groups.items():
        if not isinstance(group, Mapping):
            continue
        weights = group.get("weights")
        if not isinstance(weights, Mapping) or weights.get("num_bits") is None:
            continue
        bit_sources.append(
            (
                _positive_int("quantization weights.num_bits", weights.get("num_bits")),
                f"{source}.config_groups.{name}.weights.num_bits",
            )
        )
    bit_widths = {bits for bits, _ in bit_sources}
    if len(bit_widths) > 1:
        return None, source
    return bit_sources[0] if bit_sources else (None, source)


def _head_dim(
    config: Mapping[str, Any],
    source: str,
    hidden_size: int,
    attention_heads: int,
) -> tuple[int, str]:
    for name in ("head_dim", "v_head_dim"):
        if config.get(name) is not None:
            return _positive_int(name, config.get(name)), f"{source}.{name}"
    linear_attention = config.get("linear_attn_config")
    if isinstance(linear_attention, Mapping) and linear_attention.get("head_dim") is not None:
        return (
            _positive_int("linear_attn_config.head_dim", linear_attention.get("head_dim")),
            f"{source}.linear_attn_config.head_dim",
        )
    if hidden_size % attention_heads:
        raise ContractError("hidden_size must be divisible by num_attention_heads when head_dim is absent")
    return hidden_size // attention_heads, f"derived:{source}.hidden_size/{source}.num_attention_heads"


def _attention_type(config: Mapping[str, Any], kv_heads: int, attention_heads: int) -> str:
    if isinstance(config.get("linear_attn_config"), Mapping):
        return "hybrid"
    custom_attention = any(key in config for key in ("kv_lora_rank", "q_lora_rank", "cache_group_config"))
    if "kv_lora_rank" in config:
        return "mla"
    if custom_attention:
        return "custom"
    if kv_heads == attention_heads:
        return "mha"
    if kv_heads == 1:
        return "mqa"
    return "gqa"


def _required_manifest_inputs(
    *,
    parameter_count: object,
    attention_type: str,
    quantization_present: bool,
    quantization_bits: int | None,
    kv_bytes_per_token_per_device_override: int | None,
    resident_weight_bytes_override: int | None,
) -> list[str]:
    required: list[str] = []
    if parameter_count is None:
        required.append("parameter_count_override (logical model parameter count)")
    if attention_type in {"custom", "hybrid", "mla"} and kv_bytes_per_token_per_device_override is None:
        required.append("kv_bytes_per_token_per_device_override (measured runtime KV layout)")
    if quantization_present and quantization_bits not in (4, 8) and resident_weight_bytes_override is None:
        required.append("resident_weight_bytes_override (unsupported or mixed quantization layout)")
    return required


def _normalize_dtype(value: object) -> str:
    raw = str(value).lower()
    aliases = {"bfloat16": "bf16", "float16": "fp16", "float8": "fp8", "float8_e4m3fn": "fp8"}
    normalized = aliases.get(raw, raw)
    if normalized not in DTYPE_BITS:
        raise ContractError(f"unsupported manifest weight dtype {value!r}")
    return normalized


def _pinned_revision(revision: str) -> str:
    normalized = revision.lower()
    if len(normalized) != 40 or any(character not in "0123456789abcdef" for character in normalized):
        raise ContractError("Hugging Face revision must be an immutable 40-character commit SHA")
    return normalized


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Replayable model facts resolved from one immutable artifact revision."""

    schema_version: str
    model: ModelSpec
    artifact_bytes: int
    tensor_element_count: int | None
    field_sources: Mapping[str, str]
    evidence: tuple[EvidenceRecord, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "model-manifest-1.0":
            raise ContractError("unsupported model manifest schema_version")
        _pinned_revision(self.model.revision)
        _positive_int("artifact_bytes", self.artifact_bytes)
        if self.tensor_element_count is not None:
            _positive_int("tensor_element_count", self.tensor_element_count)
        if not self.field_sources or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.field_sources.items()
        ):
            raise ContractError("model manifest field_sources must contain non-empty string entries")
        if not self.evidence:
            raise ContractError("model manifest requires provenance evidence")
        required_sources = {
            "model_id",
            "revision",
            "parameter_count",
            "num_layers",
            "num_kv_heads",
            "head_dim",
            "weight_dtype",
            "max_model_len",
            "attention_type",
            "artifact_bytes",
        }
        missing_sources = required_sources.difference(self.field_sources)
        if missing_sources:
            raise ContractError("model manifest is missing field provenance: " + ", ".join(sorted(missing_sources)))
        object.__setattr__(self, "field_sources", dict(self.field_sources))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "artifact_bytes": self.artifact_bytes,
            "tensor_element_count": self.tensor_element_count,
            "field_sources": dict(self.field_sources),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelManifest:
        raw_sources = data.get("field_sources")
        raw_evidence = data.get("evidence")
        if data.get("schema_version") != "model-manifest-1.0":
            raise ContractError("unsupported model manifest schema_version")
        if not isinstance(raw_sources, Mapping):
            raise ContractError("field_sources must be an object")
        if not isinstance(raw_evidence, list) or not all(isinstance(item, Mapping) for item in raw_evidence):
            raise ContractError("evidence must be an array of objects")
        model_data = data.get("model")
        if not isinstance(model_data, Mapping):
            raise ContractError("model must be an object")
        return cls(
            "model-manifest-1.0",
            ModelSpec.from_dict(model_data),
            _positive_int("artifact_bytes", data.get("artifact_bytes")),
            (
                None
                if data.get("tensor_element_count") is None
                else _positive_int("tensor_element_count", data.get("tensor_element_count"))
            ),
            {str(key): _required_string("field source", value) for key, value in raw_sources.items()},
            tuple(EvidenceRecord.from_dict(item) for item in raw_evidence),
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> ModelManifest:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ContractError("model manifest must contain an object")
        return cls.from_dict(value)


def import_huggingface_manifest(
    repo_id: str,
    revision: str,
    config: Mapping[str, Any],
    safetensors_index: Mapping[str, Any],
    *,
    kv_bytes_per_token_per_device_override: int | None = None,
    parameter_count_override: int | None = None,
    safetensors_headers: Mapping[str, bytes | Mapping[str, Any]] | None = None,
    resident_weight_bytes_override: int | None = None,
) -> ModelManifest:
    """Build an offline manifest from caller-fetched Hugging Face metadata.

    No network call occurs here. Callers may fetch `config.json` and the
    SafeTensors index themselves, then cache this normalized result.
    """

    repo = _required_string("repo_id", repo_id)
    pinned = _pinned_revision(revision)
    text, text_source = _text_config(config)
    layers, layers_source = _config_int(text, "num_hidden_layers", "n_layer", source=text_source)
    kv_heads, kv_source = _config_int(
        text,
        "num_key_value_heads",
        "n_head_kv",
        "num_attention_heads",
        source=text_source,
    )
    attention_heads, _ = _config_int(text, "num_attention_heads", "n_head", source=text_source)
    hidden_size, _ = _config_int(text, "hidden_size", "n_embd", source=text_source)
    head_dim, head_source = _head_dim(text, text_source, hidden_size, attention_heads)
    max_model_len, context_source = _config_int(
        text,
        "max_position_embeddings",
        "model_max_length",
        "n_positions",
        source=text_source,
    )
    quantization, quantization_source = _quantization_config(config)
    quantization_bits: int | None = None
    dtype_source: str
    if quantization is None or quantization_source is None:
        dtype_key = "torch_dtype" if "torch_dtype" in text else "dtype" if "dtype" in text else "default"
        dtype = _normalize_dtype(text.get("torch_dtype", text.get("dtype", "bf16")))
        dtype_source = f"{text_source}.{dtype_key}"
    else:
        quantization_bits, dtype_source = _quantization_bits(quantization, quantization_source)
        dtype = f"int{quantization_bits}" if quantization_bits in (4, 8) else "quantized"
    metadata = safetensors_index.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractError("SafeTensors index is missing metadata")
    weight_bytes = _positive_int("safetensors metadata.total_size", metadata.get("total_size"))
    parameter_count_raw = (
        parameter_count_override if parameter_count_override is not None else config.get("num_parameters")
    )
    tensor_element_count = (
        None if safetensors_headers is None else tensor_element_count_from_safetensors_headers(safetensors_headers)
    )
    attention_type = _attention_type(text, kv_heads, attention_heads)
    required_inputs = _required_manifest_inputs(
        parameter_count=parameter_count_raw,
        attention_type=attention_type,
        quantization_present=quantization is not None,
        quantization_bits=quantization_bits,
        kv_bytes_per_token_per_device_override=kv_bytes_per_token_per_device_override,
        resident_weight_bytes_override=resident_weight_bytes_override,
    )
    if required_inputs:
        raise ContractError("model manifest requires caller inputs: " + "; ".join(required_inputs))
    assert parameter_count_raw is not None
    parameter_count = _positive_int("parameter_count", parameter_count_raw)
    architecture_value = config.get("architectures")
    architecture_source = "config.architectures[0]"
    if not isinstance(architecture_value, list) or not architecture_value:
        architecture_value = text.get("architectures")
        architecture_source = f"{text_source}.architectures[0]"
    architecture: str | None = None
    if isinstance(architecture_value, list) and architecture_value and isinstance(architecture_value[0], str):
        architecture = architecture_value[0]
    model = ModelSpec(
        model_id=repo,
        revision=pinned,
        parameter_count=parameter_count,
        num_layers=layers,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        weight_dtype=dtype,
        max_model_len=max_model_len,
        explicit_weight_bytes=(
            None
            if resident_weight_bytes_override is None
            else _positive_int("resident_weight_bytes_override", resident_weight_bytes_override)
        ),
        attention_type=attention_type,
        kv_bytes_per_token_per_device_override=kv_bytes_per_token_per_device_override,
        architecture=architecture,
    )
    field_sources = {
        "model_id": "caller.repo_id",
        "revision": "caller.immutable_revision",
        "parameter_count": (
            "caller.parameter_count_override" if parameter_count_override is not None else "config.num_parameters"
        ),
        "num_layers": layers_source,
        "num_kv_heads": kv_source,
        "head_dim": head_source,
        "weight_dtype": dtype_source,
        "max_model_len": context_source,
        "attention_type": f"derived:{text_source}-attention-fields",
        "artifact_bytes": "safetensors-index.metadata.total_size",
    }
    if architecture is not None:
        field_sources["architecture"] = architecture_source
    if resident_weight_bytes_override is not None:
        field_sources["explicit_weight_bytes"] = "caller.measured-resident-weight-override"
    if tensor_element_count is not None:
        field_sources["tensor_element_count"] = "safetensors-headers.tensor-shapes"
    if kv_bytes_per_token_per_device_override is not None:
        field_sources["kv_bytes_per_token_per_device_override"] = "caller.measured-runtime-override"
    evidence = EvidenceRecord(
        evidence_id=f"hf-manifest-{pinned}",
        kind=EvidenceKind.REPORTED,
        source=f"hf://{repo}@{pinned}",
        scope="config.json and SafeTensors index metadata",
        metrics={"revision": pinned, "artifact_bytes": weight_bytes},
    )
    return ModelManifest("model-manifest-1.0", model, weight_bytes, tensor_element_count, field_sources, (evidence,))


def parse_safetensors_header(data: bytes) -> Mapping[str, Any]:
    """Parse a SafeTensors header prefix without reading tensor payloads."""

    if len(data) < 8:
        raise ContractError("SafeTensors header is shorter than its 8-byte length prefix")
    header_length = struct.unpack("<Q", data[:8])[0]
    if header_length <= 0 or header_length > 16 * 1024 * 1024:
        raise ContractError("SafeTensors header length must be in (0, 16 MiB]")
    if len(data) < 8 + header_length:
        raise ContractError("SafeTensors header bytes are incomplete")
    try:
        value = json.loads(data[8 : 8 + header_length])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("SafeTensors header is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ContractError("SafeTensors header must contain an object")
    return value


def tensor_element_count_from_safetensors_headers(
    headers: Mapping[str, bytes | Mapping[str, Any]],
) -> int:
    """Count stored tensor elements from one or more SafeTensors headers."""

    if not headers:
        raise ContractError("at least one SafeTensors header is required")
    total = 0
    for shard, raw_header in headers.items():
        header = parse_safetensors_header(raw_header) if isinstance(raw_header, bytes) else raw_header
        if not isinstance(header, Mapping):
            raise ContractError(f"SafeTensors header {shard!r} must be an object")
        for name, raw_tensor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(raw_tensor, Mapping):
                raise ContractError(f"tensor {name!r} in {shard!r} must be an object")
            shape = raw_tensor.get("shape")
            if not isinstance(shape, list):
                raise ContractError(f"tensor {name!r} in {shard!r} is missing a shape array")
            elements = 1
            for dimension in shape:
                elements *= _positive_int(f"tensor {name} dimension", dimension)
            total += elements
    return _positive_int("SafeTensors tensor element count", total)


class JSONFetcher(Protocol):
    def __call__(self, url: str) -> Mapping[str, Any]: ...


class RangeFetcher(Protocol):
    def __call__(self, url: str, start: int, end: int) -> bytes: ...


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "inference-capacity-contract/0.5"})
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - fixed HTTPS host below
        _validate_huggingface_url(response.geturl())
        body = response.read(16 * 1024 * 1024 + 1)
    if len(body) > 16 * 1024 * 1024:
        raise ContractError("Hugging Face metadata response exceeds 16 MiB")
    value = json.loads(body)
    if not isinstance(value, Mapping):
        raise ContractError("Hugging Face metadata must contain a JSON object")
    return value


def _validate_huggingface_url(url: str) -> None:
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    allowed = hostname == "huggingface.co" or hostname.endswith(".huggingface.co") or hostname.endswith(".hf.co")
    if parsed.scheme != "https" or not allowed:
        raise ContractError("Hugging Face resolver followed a redirect to an untrusted host")


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "inference-capacity-contract/0.5", "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - fixed HTTPS host below
        _validate_huggingface_url(response.geturl())
        body = response.read(end - start + 2)
    expected = end - start + 1
    if len(body) != expected:
        raise ContractError(f"range response returned {len(body)} bytes; expected {expected}")
    return bytes(body)


def _fetch_safetensors_header(fetch: RangeFetcher, url: str) -> bytes:
    prefix = fetch(url, 0, 7)
    if len(prefix) != 8:
        raise ContractError("SafeTensors length prefix must contain 8 bytes")
    header_length = struct.unpack("<Q", prefix)[0]
    if header_length <= 0 or header_length > 16 * 1024 * 1024:
        raise ContractError("SafeTensors header length must be in (0, 16 MiB]")
    return prefix + fetch(url, 8, 8 + header_length - 1)


def _single_safetensors_index(header: bytes) -> Mapping[str, Any]:
    document = parse_safetensors_header(header)
    weight_map: dict[str, str] = {}
    payload_bytes = 0
    for name, value in document.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, Mapping):
            raise ContractError(f"SafeTensors tensor {name!r} must be an object")
        offsets = value.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in offsets)
            or offsets[0] > offsets[1]
        ):
            raise ContractError(f"SafeTensors tensor {name!r} has invalid data_offsets")
        payload_bytes = max(payload_bytes, offsets[1])
        weight_map[str(name)] = "model.safetensors"
    if not weight_map or payload_bytes <= 0:
        raise ContractError("model.safetensors contains no tensor payloads")
    return {"metadata": {"total_size": payload_bytes}, "weight_map": weight_map}


def resolve_huggingface_manifest(
    repo_id: str,
    revision: str,
    *,
    fetch_json: JSONFetcher | None = None,
    fetch_range: RangeFetcher | None = None,
    cache_dir: Path | None = None,
    kv_bytes_per_token_per_device_override: int | None = None,
    parameter_count_override: int | None = None,
    resident_weight_bytes_override: int | None = None,
    inspect_safetensors_headers: bool = True,
) -> ModelManifest:
    """Resolve pinned metadata without downloading model weight shards."""

    repo = _required_string("repo_id", repo_id)
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
        raise ContractError("repo_id must be in owner/name form")
    pinned = _pinned_revision(revision)
    if kv_bytes_per_token_per_device_override is not None:
        _positive_int(
            "kv_bytes_per_token_per_device_override",
            kv_bytes_per_token_per_device_override,
        )
    if parameter_count_override is not None:
        _positive_int("parameter_count_override", parameter_count_override)
    if resident_weight_bytes_override is not None:
        _positive_int("resident_weight_bytes_override", resident_weight_bytes_override)
    cache_path = None
    if cache_dir is not None:
        cache_inputs = {
            "cache_version": "model-manifest-cache-1.3",
            "repo_id": repo,
            "revision": pinned,
            "kv_bytes_per_token_per_device_override": kv_bytes_per_token_per_device_override,
            "parameter_count_override": parameter_count_override,
            "resident_weight_bytes_override": resident_weight_bytes_override,
            "inspect_safetensors_headers": inspect_safetensors_headers,
        }
        cache_digest = hashlib.sha256(
            json.dumps(cache_inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        cache_path = cache_dir / f"{repo.replace('/', '--')}--{pinned}--{cache_digest}.json"
        if cache_path.exists():
            cached = ModelManifest.read(cache_path)
            if cached.model.model_id != repo or cached.model.revision != pinned:
                raise ContractError("cached model manifest identity does not match its cache key")
            return cached
    fetch = _fetch_json if fetch_json is None else fetch_json
    root = f"https://huggingface.co/{quote(repo, safe='/')}/resolve/{pinned}"
    config = fetch(f"{root}/config.json")
    resolved_parameter_count = parameter_count_override
    api_parameter_count: int | None = None
    text, text_source = _text_config(config)
    quantization, quantization_source = _quantization_config(config)
    quantization_bits = (
        None
        if quantization is None or quantization_source is None
        else _quantization_bits(quantization, quantization_source)[0]
    )
    kv_heads, _ = _config_int(
        text,
        "num_key_value_heads",
        "n_head_kv",
        "num_attention_heads",
        source=text_source,
    )
    attention_heads, _ = _config_int(text, "num_attention_heads", "n_head", source=text_source)
    attention_type = _attention_type(text, kv_heads, attention_heads)
    if resolved_parameter_count is None and config.get("num_parameters") is None:
        if quantization is None:
            model_info = fetch(f"https://huggingface.co/api/models/{quote(repo, safe='/')}/revision/{pinned}")
            if str(model_info.get("sha", "")).lower() != pinned:
                raise ContractError("Hugging Face model metadata does not match the pinned revision")
            safetensors_info = model_info.get("safetensors")
            if not isinstance(safetensors_info, Mapping):
                raise ContractError("Hugging Face model metadata is missing SafeTensors parameter totals")
            api_parameter_count = _positive_int("safetensors.total", safetensors_info.get("total"))
            resolved_parameter_count = api_parameter_count
    required_inputs = _required_manifest_inputs(
        parameter_count=(
            resolved_parameter_count if resolved_parameter_count is not None else config.get("num_parameters")
        ),
        attention_type=attention_type,
        quantization_present=quantization is not None,
        quantization_bits=quantization_bits,
        kv_bytes_per_token_per_device_override=kv_bytes_per_token_per_device_override,
        resident_weight_bytes_override=resident_weight_bytes_override,
    )
    if required_inputs:
        raise ContractError("model manifest requires caller inputs: " + "; ".join(required_inputs))
    try:
        index: Mapping[str, Any] | None = fetch(f"{root}/model.safetensors.index.json")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        index = None
    except FileNotFoundError:
        index = None
    range_fetch = _fetch_range if fetch_range is None else fetch_range
    headers: dict[str, bytes] | None = None
    single_file = index is None
    if index is None:
        if not inspect_safetensors_headers:
            raise ContractError("a single-file SafeTensors model requires header inspection")
        shard = "model.safetensors"
        header = _fetch_safetensors_header(range_fetch, f"{root}/{shard}")
        index = _single_safetensors_index(header)
        headers = {shard: header}
    elif inspect_safetensors_headers:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ContractError("SafeTensors index weight_map is required to resolve tensor headers")
        shards = sorted({value for value in weight_map.values() if isinstance(value, str) and value})
        if not shards:
            raise ContractError("SafeTensors index weight_map contains no shard names")
        headers = {}
        for shard in shards:
            shard_url = f"{root}/{quote(shard, safe='.-_')}"
            headers[shard] = _fetch_safetensors_header(range_fetch, shard_url)
    assert index is not None
    manifest = import_huggingface_manifest(
        repo,
        pinned,
        config,
        index,
        kv_bytes_per_token_per_device_override=kv_bytes_per_token_per_device_override,
        parameter_count_override=resolved_parameter_count,
        safetensors_headers=headers,
        resident_weight_bytes_override=resident_weight_bytes_override,
    )
    field_sources = dict(manifest.field_sources)
    evidence = manifest.evidence
    if single_file:
        field_sources["artifact_bytes"] = "model.safetensors.header.data_offsets"
        evidence = tuple(replace(item, scope="config.json and model.safetensors header metadata") for item in evidence)
    if api_parameter_count is not None:
        field_sources["parameter_count"] = "huggingface-api.safetensors.total"
        evidence = (
            *evidence,
            EvidenceRecord(
                evidence_id=f"hf-model-info-{pinned}",
                kind=EvidenceKind.REPORTED,
                source=f"https://huggingface.co/api/models/{repo}/revision/{pinned}",
                scope="pinned model metadata",
                metrics={"revision": pinned, "parameter_count": api_parameter_count},
            ),
        )
    if single_file or api_parameter_count is not None:
        manifest = replace(manifest, field_sources=field_sources, evidence=evidence)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.write(cache_path)
    return manifest


@dataclass(frozen=True, slots=True)
class VLLMInitializationProfile:
    schema_version: str
    model_revision: str
    hardware_id: str
    runtime_engine: str
    runtime_version: str
    tensor_parallel_size: int
    data_parallel_size: int
    expert_parallel_size: int
    physical_device_count: int
    memory_utilization_limit: float
    kv_cache_dtype: str
    block_size_tokens: int
    weight_bytes_per_device: int
    runtime_overhead_bytes_per_device: int
    activation_reserve_bytes_per_device: int
    kv_bytes_per_token_per_device: int
    kv_capacity_tokens_per_device: int
    evidence: EvidenceRecord

    def __post_init__(self) -> None:
        if self.schema_version != "vllm-initialization-profile-1.0":
            raise ContractError("unsupported vLLM initialization profile schema_version")
        for name in ("model_revision", "hardware_id", "runtime_engine", "runtime_version", "kv_cache_dtype"):
            _required_string(name, getattr(self, name))
        if self.kv_cache_dtype not in DTYPE_BITS:
            raise ContractError("unsupported initialization KV cache dtype")
        for name in (
            "tensor_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
            "physical_device_count",
            "block_size_tokens",
            "weight_bytes_per_device",
            "kv_bytes_per_token_per_device",
            "kv_capacity_tokens_per_device",
        ):
            _positive_int(name, getattr(self, name))
        for name in ("runtime_overhead_bytes_per_device", "activation_reserve_bytes_per_device"):
            _non_negative_int(name, getattr(self, name))
        utilization = _positive_number("memory_utilization_limit", self.memory_utilization_limit)
        if utilization > 1:
            raise ContractError("memory_utilization_limit must be in (0, 1]")
        engine_ranks = self.tensor_parallel_size * self.data_parallel_size
        if engine_ranks > self.physical_device_count:
            raise ContractError("initialization TP x DP exceeds physical_device_count")
        if engine_ranks % self.expert_parallel_size:
            raise ContractError("initialization expert_parallel_size must divide TP x DP")
        if self.evidence.kind != EvidenceKind.MEASURED:
            raise ContractError("vLLM initialization profile requires measured evidence")
        claim = {
            "model_revision": self.model_revision,
            "hardware_id": self.hardware_id,
            "runtime_engine": self.runtime_engine,
            "runtime_version": self.runtime_version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "physical_device_count": self.physical_device_count,
            "memory_utilization_limit": self.memory_utilization_limit,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size_tokens": self.block_size_tokens,
            "weight_bytes_per_device": self.weight_bytes_per_device,
            "runtime_overhead_bytes_per_device": self.runtime_overhead_bytes_per_device,
            "activation_reserve_bytes_per_device": self.activation_reserve_bytes_per_device,
            "kv_bytes_per_token_per_device": self.kv_bytes_per_token_per_device,
            "kv_capacity_tokens_per_device": self.kv_capacity_tokens_per_device,
        }
        if any(self.evidence.metrics.get(name) != value for name, value in claim.items()):
            raise ContractError("initialization evidence must bind its exact identity and every memory metric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "hardware_id": self.hardware_id,
            "runtime_engine": self.runtime_engine,
            "runtime_version": self.runtime_version,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "physical_device_count": self.physical_device_count,
            "memory_utilization_limit": self.memory_utilization_limit,
            "kv_cache_dtype": self.kv_cache_dtype,
            "block_size_tokens": self.block_size_tokens,
            "weight_bytes_per_device": self.weight_bytes_per_device,
            "runtime_overhead_bytes_per_device": self.runtime_overhead_bytes_per_device,
            "activation_reserve_bytes_per_device": self.activation_reserve_bytes_per_device,
            "kv_bytes_per_token_per_device": self.kv_bytes_per_token_per_device,
            "kv_capacity_tokens_per_device": self.kv_capacity_tokens_per_device,
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VLLMInitializationProfile:
        evidence = data.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ContractError("initialization evidence must be an object")
        return cls(
            schema_version=_required_string("schema_version", data.get("schema_version")),
            model_revision=_required_string("model_revision", data.get("model_revision")),
            hardware_id=_required_string("hardware_id", data.get("hardware_id")),
            runtime_engine=_required_string("runtime_engine", data.get("runtime_engine")),
            runtime_version=_required_string("runtime_version", data.get("runtime_version")),
            tensor_parallel_size=_positive_int("tensor_parallel_size", data.get("tensor_parallel_size")),
            data_parallel_size=_positive_int("data_parallel_size", data.get("data_parallel_size")),
            expert_parallel_size=_positive_int("expert_parallel_size", data.get("expert_parallel_size")),
            physical_device_count=_positive_int("physical_device_count", data.get("physical_device_count")),
            memory_utilization_limit=_positive_number("memory_utilization_limit", data.get("memory_utilization_limit")),
            kv_cache_dtype=_required_string("kv_cache_dtype", data.get("kv_cache_dtype")),
            block_size_tokens=_positive_int("block_size_tokens", data.get("block_size_tokens")),
            weight_bytes_per_device=_positive_int("weight_bytes_per_device", data.get("weight_bytes_per_device")),
            runtime_overhead_bytes_per_device=_non_negative_int(
                "runtime_overhead_bytes_per_device", data.get("runtime_overhead_bytes_per_device")
            ),
            activation_reserve_bytes_per_device=_non_negative_int(
                "activation_reserve_bytes_per_device", data.get("activation_reserve_bytes_per_device")
            ),
            kv_bytes_per_token_per_device=_positive_int(
                "kv_bytes_per_token_per_device", data.get("kv_bytes_per_token_per_device")
            ),
            kv_capacity_tokens_per_device=_positive_int(
                "kv_capacity_tokens_per_device", data.get("kv_capacity_tokens_per_device")
            ),
            evidence=EvidenceRecord.from_dict(evidence),
        )


def import_vllm_initialization(data: Mapping[str, Any], *, source: str) -> VLLMInitializationProfile:
    """Normalize measured per-device memory facts from a vLLM initialization run."""

    source_value = _required_string("source", source)
    revision = _required_string("model_revision", data.get("model_revision"))
    identities: dict[str, str | int | float] = {
        "hardware_id": _required_string("hardware_id", data.get("hardware_id")),
        "runtime_engine": _required_string("runtime_engine", data.get("runtime_engine")),
        "runtime_version": _required_string("runtime_version", data.get("runtime_version")),
        "tensor_parallel_size": _positive_int("tensor_parallel_size", data.get("tensor_parallel_size")),
        "data_parallel_size": _positive_int("data_parallel_size", data.get("data_parallel_size")),
        "expert_parallel_size": _positive_int("expert_parallel_size", data.get("expert_parallel_size")),
        "physical_device_count": _positive_int("physical_device_count", data.get("physical_device_count")),
        "memory_utilization_limit": _positive_number("memory_utilization_limit", data.get("memory_utilization_limit")),
        "kv_cache_dtype": _required_string("kv_cache_dtype", data.get("kv_cache_dtype")),
        "block_size_tokens": _positive_int("block_size_tokens", data.get("block_size_tokens")),
    }
    if float(identities["memory_utilization_limit"]) > 1:
        raise ContractError("memory_utilization_limit must be in (0, 1]")
    fields = {
        "weight_bytes_per_device": _positive_int("weight_bytes_per_device", data.get("weight_bytes_per_device")),
        "runtime_overhead_bytes_per_device": _non_negative_int(
            "runtime_overhead_bytes_per_device", data.get("runtime_overhead_bytes_per_device")
        ),
        "activation_reserve_bytes_per_device": _non_negative_int(
            "activation_reserve_bytes_per_device", data.get("activation_reserve_bytes_per_device")
        ),
        "kv_bytes_per_token_per_device": _positive_int(
            "kv_bytes_per_token_per_device", data.get("kv_bytes_per_token_per_device")
        ),
        "kv_capacity_tokens_per_device": _positive_int(
            "kv_capacity_tokens_per_device", data.get("kv_capacity_tokens_per_device")
        ),
    }
    evidence = EvidenceRecord(
        evidence_id="vllm-initialization-profile",
        kind=EvidenceKind.MEASURED,
        source=source_value,
        scope=f"vLLM initialization for model revision {revision}",
        metrics={"model_revision": revision, **identities, **fields},
    )
    return VLLMInitializationProfile(
        "vllm-initialization-profile-1.0",
        revision,
        str(identities["hardware_id"]),
        str(identities["runtime_engine"]),
        str(identities["runtime_version"]),
        int(identities["tensor_parallel_size"]),
        int(identities["data_parallel_size"]),
        int(identities["expert_parallel_size"]),
        int(identities["physical_device_count"]),
        float(identities["memory_utilization_limit"]),
        str(identities["kv_cache_dtype"]),
        int(identities["block_size_tokens"]),
        fields["weight_bytes_per_device"],
        fields["runtime_overhead_bytes_per_device"],
        fields["activation_reserve_bytes_per_device"],
        fields["kv_bytes_per_token_per_device"],
        fields["kv_capacity_tokens_per_device"],
        evidence,
    )


def materialize_recipe_draft(
    draft: RecipeDraft,
    manifest: ModelManifest,
    initialization: VLLMInitializationProfile,
    *,
    precise_prefix_routing: bool,
) -> ServingRecipe:
    """Complete a draft with one pinned manifest and matching initialization run."""

    draft_model_id = recipe_string(draft, "model.model_id")
    if manifest.model.model_id != draft_model_id:
        raise ContractError("model manifest identity does not match the recipe draft")
    if initialization.model_revision != manifest.model.revision:
        raise ContractError("initialization model revision does not match the model manifest")
    configured_max_model_len = recipe_integer(draft, "model.max_model_len")
    manifest_max_model_len = manifest.model.max_model_len
    if manifest_max_model_len is None:
        raise ContractError("model manifest is missing max_model_len")
    if configured_max_model_len > manifest_max_model_len:
        raise ContractError("recipe max_model_len exceeds the model manifest limit")
    expected_identity: dict[str, str | int | float] = {
        "hardware_id": recipe_string(draft, "host.hardware_id"),
        "runtime_engine": recipe_string(draft, "runtime.engine"),
        "runtime_version": recipe_string(draft, "runtime.version"),
        "tensor_parallel_size": recipe_integer(draft, "topology.tensor_parallel_size"),
        "data_parallel_size": recipe_integer(draft, "topology.data_parallel_size"),
        "expert_parallel_size": recipe_integer(draft, "topology.expert_parallel_size"),
        "physical_device_count": recipe_integer(draft, "topology.physical_device_count"),
        "memory_utilization_limit": recipe_number(draft, "host.memory_utilization_limit"),
        "kv_cache_dtype": recipe_string(draft, "runtime.kv_cache_dtype"),
        "block_size_tokens": recipe_integer(draft, "runtime.block_size_tokens"),
    }
    for name, expected in expected_identity.items():
        if getattr(initialization, name) != expected:
            raise ContractError(f"initialization {name} does not match the recipe draft")
    document_evidence = draft.document.get("evidence")
    if not isinstance(document_evidence, list):
        raise ContractError("recipe draft evidence must be an array")
    evidence = [*document_evidence, *(item.to_dict() for item in manifest.evidence), initialization.evidence.to_dict()]
    overrides = {
        "model": {
            **manifest.model.to_dict(),
            "max_model_len": configured_max_model_len,
            "kv_bytes_per_token_per_device_override": initialization.kv_bytes_per_token_per_device,
        },
        "runtime": {
            "weight_bytes_per_device_override": initialization.weight_bytes_per_device,
            "runtime_overhead_bytes_per_device": initialization.runtime_overhead_bytes_per_device,
            "activation_reserve_bytes_per_device": initialization.activation_reserve_bytes_per_device,
        },
        "llmd": {"precise_prefix_routing": precise_prefix_routing},
        "evidence": evidence,
    }
    recipe = draft.to_serving_recipe(overrides)
    rank_hardware = replace(recipe.host, device_count=recipe.topology.tensor_parallel_size)
    contract = capacity_for(recipe.model, rank_hardware, recipe.runtime)
    if contract.kv_capacity_tokens_per_device != initialization.kv_capacity_tokens_per_device:
        raise ContractError(
            "initialization KV capacity does not reconcile with weights, reserves, KV bytes/token, and block size"
        )
    return recipe


def recipe_value(draft: RecipeDraft, path: str) -> object:
    value: object = draft.document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value or value[component] is None:
            raise ContractError(f"recipe draft is missing {path}")
        value = value[component]
    return value


def recipe_string(draft: RecipeDraft, path: str) -> str:
    value = recipe_value(draft, path)
    if not isinstance(value, str) or not value:
        raise ContractError(f"recipe draft {path} must be a non-empty string")
    return value


def recipe_integer(draft: RecipeDraft, path: str) -> int:
    return _positive_int(path, recipe_value(draft, path))


def recipe_number(draft: RecipeDraft, path: str) -> float:
    return _positive_number(path, recipe_value(draft, path))


def import_vllm_benchmark(
    recipe: ServingRecipe,
    data: Mapping[str, Any],
    *,
    context_tokens: int,
    latency_percentile: float,
    source: str,
    concurrent_sequences: int | None = None,
) -> MeasuredGroupProfile:
    """Normalize aggregate `vllm bench serve` counters into one exact claim."""

    completed = _positive_int("completed", data.get("completed"))
    duration = _positive_number("duration", data.get("duration"))
    input_tokens = _positive_int("total_input_tokens", data.get("total_input_tokens"))
    output_tokens = _positive_int("total_output_tokens", data.get("total_output_tokens"))
    context = _positive_int("context_tokens", context_tokens)
    source_value = _required_string("source", source)
    percentile = _positive_number("latency_percentile", latency_percentile)
    if percentile >= 100:
        raise ContractError("latency_percentile must be below 100")
    percentile_label = str(int(percentile)) if percentile.is_integer() else str(percentile).replace(".", "_")
    ttft_key = f"p{percentile_label}_ttft_ms"
    tpot_key = f"p{percentile_label}_tpot_ms"
    if ttft_key not in data:
        raise ContractError(f"benchmark is missing {ttft_key}")
    ttft = _positive_number(ttft_key, data[ttft_key])
    tpot = _positive_number(tpot_key, data[tpot_key]) if tpot_key in data else None
    if concurrent_sequences is not None:
        _positive_int("concurrent_sequences", concurrent_sequences)
    requests_per_second = completed / duration
    input_per_request = input_tokens / completed
    output_per_request = output_tokens / completed
    prefill_tps = input_tokens / duration
    decode_tps = output_tokens / duration
    if input_per_request + output_per_request > context:
        raise ContractError("average input plus output tokens exceeds context_tokens")
    for name, expected in (
        ("request_throughput", requests_per_second),
        ("input_throughput", prefill_tps),
        ("output_throughput", decode_tps),
    ):
        if name in data and not isclose(_positive_number(name, data[name]), expected, rel_tol=1e-6, abs_tol=1e-9):
            raise ContractError(f"benchmark {name} conflicts with aggregate counters")
    metrics: dict[str, float | int | str] = {
        "recipe_variant_fingerprint_version": RECIPE_VARIANT_FINGERPRINT_VERSION,
        "recipe_variant_fingerprint": recipe.variant_fingerprint,
        "context_tokens": context,
        "input_tokens_per_request": input_per_request,
        "output_tokens_per_request": output_per_request,
        "requests_per_second": requests_per_second,
        "prefill_tokens_per_second": prefill_tps,
        "decode_tokens_per_second": decode_tps,
        "ttft_ms": ttft,
        "latency_percentile": percentile,
    }
    if concurrent_sequences is not None:
        metrics["concurrent_sequences"] = concurrent_sequences
    if tpot is not None:
        metrics["tpot_ms"] = tpot
    evidence = EvidenceRecord(
        evidence_id="vllm-bench-serve-operating-point",
        kind=EvidenceKind.MEASURED,
        source=source_value,
        scope="exact recipe fingerprint and aggregate benchmark operating point",
        metrics=metrics,
    )
    return MeasuredGroupProfile(
        profile_id="vllm-bench-serve-profile",
        recipe_variant_fingerprint_version=RECIPE_VARIANT_FINGERPRINT_VERSION,
        recipe_variant_fingerprint=recipe.variant_fingerprint,
        context_tokens=context,
        input_tokens_per_request=input_per_request,
        output_tokens_per_request=output_per_request,
        requests_per_second=requests_per_second,
        prefill_tokens_per_second=prefill_tps,
        decode_tokens_per_second=decode_tps,
        concurrent_sequences=concurrent_sequences,
        ttft_ms=ttft,
        tpot_ms=tpot,
        latency_percentile=percentile,
        evidence=(evidence,),
    )
