# Inference Capacity Contract

`inference-capacity-contract` is a dependency-free Python library for
calculating LLM memory and KV-cache capacity. It answers this question:

> Given an exact model version, runtime configuration, and hardware layout for
> one serving replica, what fits in memory?

The result is a versioned JSON contract with:

- weight, runtime-reserve, activation-reserve, and KV-cache memory accounting;
- model, vendor, and tensor-parallel topology compatibility checks;
- per-device KV bytes/token, block budget, and token budget;
- a context-dependent concurrency envelope instead of one misleading scalar;
- assumptions, warnings, a validation level, and evidence sources;
- reverse-fit results across a hardware inventory;
- llm-d planner and scaling-policy payloads;
- measured-profile replica recommendations with cost and GPU-hour deltas;
- TP, DP, and EP serving recipe checks; and
- a direct answer for whether a configured recipe can handle the requested
  context and load.

The library does not start vLLM or SGLang, discover GPUs, resolve Hugging Face
models, query provider catalogs, mutate a cluster, or claim throughput and
latency from model parameters. Callers supply pinned model and hardware facts.
Runtime initialization and SLO performance require matching measured evidence.

## Why the contract is useful

Serving systems often collapse different claims into one capacity number:

1. whether the model fits in memory;
2. runtime initialization success;
3. measured workload/SLO performance;
4. replica policy; and
5. changes to a live cluster.

This project reports those claims separately. Its memory calculation can rule
out hardware and support planning, but it does not promise production capacity.

## Install from a checkout

Requires Python 3.12+.

```bash
git clone https://github.com/santho090/inference-capacity-contract.git
cd inference-capacity-contract
python -m pip install .
icc --help
```

The runtime package has no third-party dependencies. Development tools are
installed separately with `python -m pip install -e '.[dev]'`.

## Choose the right operation

| Question | Python API | CLI |
|---|---|---|
| Does this model fit this hardware/runtime? | `capacity_for` | `icc plan` |
| Which supplied hardware candidates fit? | `what_fits` | `icc fit` |
| How many replicas does a measured workload need? | `recommend_scale` | `icc scale` |
| Is this complete TP/DP/EP and llm-d recipe good for this load? | `audit_recipe` | `icc audit` |
| Is an existing capacity document valid? | `CapacityContract.from_dict` | `icc validate` |
| How do I pass a contract to another planner? | adapter functions | `icc export` |

## Python example

```python
from inference_capacity_contract import (
    HardwareSpec,
    ModelSpec,
    RuntimeVariant,
    capacity_for,
)

model = ModelSpec(
    model_id="example/7b",
    revision="sha256:example",
    parameter_count=7_000_000_000,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    max_model_len=8192,
)
hardware = HardwareSpec(
    hardware_id="h100-80gb",
    vendor="nvidia",
    device_count=1,
    memory_bytes_per_device=80 * 1024**3,
)
runtime = RuntimeVariant(
    engine="vllm",
    version="0.8.5",
    runtime_overhead_bytes_per_device=1 * 1024**3,
    activation_reserve_bytes_per_device=2 * 1024**3,
    max_num_seqs=128,
)

contract = capacity_for(model, hardware, runtime)
print(contract.max_sequences_at(2048))  # 128: runtime max_num_seqs cap
print(contract.max_sequences_at(8192))  # 55: KV-memory bound
```

The sequence limit depends on the active tokens in each sequence. For that
reason, `capacity-contract-2.0` reports capacity by context length instead of a
single `max_concurrent_sequences` value.

## CLI example

```bash
icc plan \
  --model docs/fixtures/model-7b.json \
  --hardware docs/fixtures/hardware-h100.json \
  --runtime docs/fixtures/runtime-vllm.json \
  --output contract.json

icc validate --contract contract.json

icc fit \
  --model docs/fixtures/model-7b.json \
  --inventory docs/fixtures/hardware-inventory.json \
  --runtime docs/fixtures/runtime-vllm.json

icc scale \
  --contract contract.json \
  --profile docs/fixtures/workload-profile.json

icc audit \
  --recipe docs/fixtures/serving-recipe-hybrid-tp8.json \
  --load docs/fixtures/load-context-concurrency.json

icc export \
  --contract contract.json \
  --target llmd-planner
```

`fit` checks every hardware and runtime pair. If a pair is unsupported, the
result includes `unsupported_reason` and the search continues.

## Audit an existing serving recipe

`audit_recipe` accepts a normalized serving recipe and a load requirement. A
recipe describes one serving group: its model, host, vLLM or SGLang settings,
TP/DP/EP layout, independent KV ranks, llm-d limits, and current group count.
The library computes a SHA-256 fingerprint over that variant. It excludes the
recipe name, configured group count, evidence, and device price. None of those
changes per-group performance, so one measured profile can be reused while
exploring group count or cost.

Runtime notes and vendor-support metadata are also excluded. Numeric fields are
canonicalized before hashing.

The fingerprint preimage is versioned as `serving-recipe-variant-1.0`.

The fixture files can also be used directly from Python:

```python
import json
from pathlib import Path

from inference_capacity_contract import LoadRequirement, ServingRecipe, audit_recipe

recipe_path = Path("docs/fixtures/serving-recipe-hybrid-tp8.json")
load_path = Path("docs/fixtures/load-context-concurrency.json")
recipe = ServingRecipe.from_dict(json.loads(recipe_path.read_text()))
load = LoadRequirement.from_dict(json.loads(load_path.read_text()))

audit = audit_recipe(recipe, load)
print(audit.status)  # sufficient
print(audit.required_groups)  # 2
print(audit.required_devices)  # 16
print(audit.additional_devices_needed)  # 0
```

The audit checks:

- whether weights, reserves, and KV cache fit on each KV rank;
- how many sequences fit at the requested context;
- how DP ranks combine into group capacity;
- whether llm-d and the runtime agree on block size, flow-control tokens, and
  concurrency; and
- how many groups and devices the requested load needs.

The result also reports the configured device count and any additional groups
or devices needed. Those shortfall fields are `null` when measurements are
missing, because the library does not have enough information to size the load.

| Status | Meaning |
|---|---|
| `sufficient` | The configured groups cover every evaluated driver. |
| `insufficient` | The recipe cannot meet the requested context, load, or latency target. |
| `incomplete` | A traffic or latency requirement has no matching measurement. |
| `invalid` | The recipe, routing settings, or attached evidence contradicts the contract. |

Context and concurrency can be checked analytically. RPS, prefill TPS, decode
TPS, TTFT, and TPOT need measurements from the exact model, runtime, and
hardware combination. The library returns `incomplete` instead of inventing a
throughput estimate.

Measurements live in `MeasuredGroupProfile`. The profile records the recipe
fingerprint, context, request shape, per-group traffic capacity, concurrency,
and observed latency. One measured evidence record must carry the same
fingerprint, context, request shape, traffic/concurrency point, and latency
values. This prevents a low-load latency run from being combined with an
unrelated high-throughput run. A profile from a different recipe or workload
shape makes the audit `invalid`.

Traffic may be given as RPS plus tokens per request, or as direct prefill and
decode TPS. If both forms are present, they must agree.

`context_tokens` means peak active tokens per sequence, including prompt tokens
and the allowed generated tokens. TTFT and TPOT comparisons also require the
same percentile in the load and measured profile.

Pipeline parallelism and prefill/decode disaggregation are not represented in
`serving-recipe-1.0`. Normalize only TP/DP/EP serving groups with one shared
capacity pool. Adapters must reject other topologies until the schema models
them explicitly.

## Memory and KV calculation

For a standard attention path, per-device KV bytes per token are:

```text
2 × layers × KV heads resident per device × head dimension × KV dtype bytes
```

Usable memory is `device_memory × memory_utilization_limit`. The contract
subtracts per-device weights, runtime overhead, and activation reserve. The
remaining memory is divided into runtime-sized KV blocks:

```text
KV blocks = floor(KV available bytes / bytes per KV block per device)
blocks per sequence = ceil(active context tokens / block size)
max sequences at context = floor(KV blocks / blocks per sequence)
```

`max_num_seqs`, when configured, further caps the sequence count.

Tensor-parallel KV layout is runtime-specific. For tensor parallelism greater
than one, the caller or future runtime adapter must provide
`kv_heads_per_device`, interpreted as the maximum resident on any one device.
Across TP devices it must cover every logical KV head; replication is allowed.
The library will not silently assume that KV heads are sharded or replicated.
MLA, hybrid cache groups, unequal K/V dimensions,
sub-byte KV formats, and custom attention require an explicit
per-device KV-bytes/token override. The closed-form calculation is only for
uniform full-attention K/V storage.

Tensor-parallel weight layouts can contain replicated tensors or uneven shards.
Without `weight_bytes_per_device_override`, the calculator divides total
artifact bytes evenly and adds a warning. Runtime adapters should provide a
measured or manifest-derived per-device value when one is available.

## Validation levels

`capacity_for` produces `analytically-feasible`. The schema also defines:

- `runtime-supported`;
- `initialization-validated`; and
- `slo-validated`.

Attaching an `EvidenceRecord` never silently promotes a contract. The producer
must perform and record the validation step that justifies a higher level.
Every generated contract records the formula and schema version along with the
exact model, hardware, and runtime used in the calculation.

## Evidence-backed scaling

`WorkloadProfile` and `recommend_scale` use measured per-replica request,
prefill, decode, or concurrency capacity. The profile must match the contract's
model revision, hardware ID, runtime engine, and runtime version.

The recommendation reports:

- required and buffered replicas;
- the limiting demand driver;
- estimated GPU-hours per hour and GPU-hour delta from a baseline;
- hourly cost and savings when a price is provided; and
- incomplete-evidence and configured-bound warnings.

Peak-concurrency inputs must include `concurrency_context_tokens` so the
context-specific KV bound is used. They must also include measured
`sustainable_concurrent_sequences_per_replica`; the lower of that measurement
and the analytical KV bound drives the plan. The result is a policy input,
never a cluster mutation.

## Schema compatibility

Version `0.2.0` introduces the breaking `capacity-contract-2.0` schema:

- removes scalar `max_concurrent_sequences`;
- separates per-device KV bytes, blocks, and tokens;
- adds `concurrency_envelope`;
- treats one runtime variant as one tensor-parallel replica; and
- moves replica/data-parallel count to scaling and future planning layers.

The historical 1.0 schema remains in `schemas/` for reference. New documents
must use `schemas/capacity-contract-2.0.schema.json` and scaling recommendations
use `schemas/scaling-recommendation-2.0.schema.json`.

Version `0.3.0` adds these independent schemas without changing the 2.0
capacity contract:

- `serving-recipe-1.0`;
- `load-requirement-1.0`; and
- `recipe-audit-1.0`.

These schemas describe normalized output documents produced by `to_dict()`.
CLI input files may omit nullable/defaulted fields; the dependency-free Python
parsers enforce their input contracts directly.

## Roadmap

The next milestone expands the library into a single-model planner:

1. resolve pinned Hugging Face artifacts and exact tensor bytes;
2. accept normalized user-supplied provider inventories;
3. compare and rank recipes across those providers;
4. import measured vLLM profiles for RPS/TPS/TTFT/TPOT planning; and
5. later add live catalogs and a separate multi-model portfolio planner.

No HTTP service or autoscaler integration precedes a validated library contract.
See `docs/roadmap.md`.

## Project status

Alpha. Treat analytical fit as a filter, not a deployment guarantee. Stability
requires real NVIDIA and AMD initialization runs, measured profile imports, and
a published prediction-error/failure table.

## License

Apache-2.0. See `LICENSE`.
