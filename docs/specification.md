# Capacity contract specification

Current schema identifier: `capacity-contract-3.0`

The JSON document is the interchange format. Python dataclasses expose the same
fields for callers using the library directly.

## Per-replica boundary

A capacity contract describes one tensor-parallel serving replica. Its hardware
device count must equal the runtime tensor-parallel size. Replica count and data
parallelism belong to the scaling/planning layer rather than the per-replica
memory calculation.

## Inputs

| Object | Required facts | Meaning |
|---|---|---|
| `model` | immutable ID/revision, parameter count, layers, KV heads, head dimension | One model artifact. Explicit resident bytes override nominal dtype arithmetic. |
| `hardware` | ID, vendor, device count, memory per device | Devices allocated to one tensor-parallel replica. |
| `runtime` | engine/version, TP size, KV dtype, block size, reserves | One pinned runtime variant. Linear TP calculations require an explicit per-device KV-head layout; exact non-uniform weight layouts use a per-device byte override. Hybrid, MLA, custom, and sub-byte caches require context-bound runtime capacity. |

For TP greater than one, `kv_heads_per_device` is the maximum number resident
on any device. The product of that value and TP size must cover all logical KV
heads. This permits replication without allowing an under-counted layout.

## Memory ledger

`memory_bytes_per_device` contains:

- `usable_budget`: device memory multiplied by the utilization limit;
- `weights`: resident weight bytes divided across TP devices and rounded up;
- `runtime_reserve`: declared non-model runtime reserve;
- `activation_reserve`: declared activation reserve; and
- `kv_available`: remaining bytes available to KV blocks.

These are analytical inputs and results. They are not a measurement of a live
runtime unless matching evidence is attached.

## KV and concurrency

`capacity-contract-3.0` has two KV capacity modes.

For uniform full self-attention with equal K/V dimensions, `linear` mode
derives per-device KV bytes/token from the layers, KV heads resident on that
device, head dimension, and KV dtype:

```text
bytes_per_block_per_device
    = kv_bytes_per_token_per_device × kv_block_size_tokens

kv_capacity_blocks_per_device
    = floor(kv_available / bytes_per_block_per_device)

blocks_per_sequence(context_tokens)
    = ceil(context_tokens / kv_block_size_tokens)

max_sequences_at(context_tokens)
    = floor(kv_capacity_blocks_per_device / blocks_per_sequence)
```

`runtime.max_num_seqs` caps the final value when configured.

MLA, hybrid cache groups, unequal K/V dimensions, sub-byte formats, and custom
attention use `context-envelope` mode. The runtime supplies exact
`{context_tokens, max_sequences}` points plus the hardware ID and available KV
memory bytes from the initialization run. Scalar bytes/token, block-capacity,
and token-capacity fields are `null`. Requests for an unrecorded context raise
`ContractError`; the library never interpolates or extrapolates these points.

`concurrency_envelope` records the selected function or exact runtime points.
In linear mode, consumers may call `max_sequences_at(context_tokens)` for any
supported point. In context-envelope mode, they may call it only for a recorded
point. The schema has no context-free maximum sequence field because the
quantity is not a scalar.

These capacities are per device because every TP device stores a shard of the
same sequences. They are already the replica bottleneck and must not be
multiplied by `hardware.device_count`.

## Compatibility and failure behavior

`fits=true` requires:

1. declared runtime support for the hardware vendor;
2. hardware device count equal to tensor-parallel size;
3. a known per-device KV layout for a linear TP calculation, or an exact
   runtime envelope for a non-linear layout; and
4. either enough usable memory for one linear KV block or an envelope whose
   hardware identity and KV memory budget match the contract.

`capacity_for` raises `ContractError` for malformed or underspecified direct
inputs. `what_fits` converts candidate-specific errors into
`unsupported_reason`, allowing the remaining inventory to be evaluated.

## Scaling recommendation

`scaling-recommendation-2.0` requires a measured `WorkloadProfile` matching the
model revision, hardware ID, runtime engine, and runtime version. Each rate
driver uses:

```text
ceil(demand / (measured per-replica capacity × target utilization))
```

The maximum driver is buffered and bounded by configured min/max replicas.
Peak concurrency also requires `concurrency_context_tokens` and a
measured `sustainable_concurrent_sequences_per_replica`. The solver uses the
lower of that measurement and the contract's context-specific KV bound.

The recommendation reports replica count, GPU-hours per hour, and GPU-hour
delta. When a price is available, it also reports hourly cost and cost delta.
The result is a policy input, not an actuation command.

## Serving recipe audit

`serving-recipe-1.0` adds the host-level topology that a per-replica contract
does not contain. `tensor_parallel_size * data_parallel_size` is the number of
engine ranks. `physical_device_count` is the number of devices allocated to
the group. `independent_kv_ranks` says how many ranks own separate KV pools.

The audit calculates one KV-rank contract using the TP device count, then
multiplies sequence capacity by the number of independent KV ranks. Linear
contracts also expose a total KV-token capacity; context-envelope contracts do
not invent one. Capacity is never multiplied by expert-parallel size. EP
changes model placement, so an EP recipe must supply resident weight bytes per
device.

For a context or concurrency-only request, the analytical memory bound can
produce a group count. Nonzero RPS, prefill TPS, or decode TPS requires the
matching measured per-group capacity. TTFT and TPOT targets require observed
values. Missing measurements produce `incomplete`, not a guessed result.

`MeasuredGroupProfile` binds those values to a SHA-256 fingerprint of the
recipe's model, host, runtime, topology, and llm-d settings. It also records the
context and request shape used by the measurement. One measured evidence record
must carry that identity and every populated traffic, concurrency, and latency
metric. This keeps latency and capacity tied to one operating point. The audit
rejects a profile when its fingerprint, context, or supplied request shape
differs from the requested recipe and load.

The fingerprint preimage includes `serving-recipe-variant-1.0`. The audit
recomputes it from the supplied recipe. Recipe names, configured group count,
evidence, and device price are excluded because they do not change per-group
performance. Runtime notes and vendor-support metadata are also excluded, and
numeric fields are canonicalized before hashing.

Demand may be supplied as RPS and tokens per request or as direct prefill and
decode TPS. When both forms are supplied, their token rates must agree.
`context_tokens` is the peak active prompt plus generated tokens per sequence.
Latency targets and observations must use the same explicit percentile.

The audit also compares llm-d block size, flow-control token limit, and maximum
concurrent sequences with the calculated runtime values. It block-aligns the
flow budget at the requested context and uses the lowest linear-memory or
runtime-envelope, runtime, llm-d, or measured-profile concurrency limit when
sizing groups. A mismatch makes the
recipe `invalid`. The result records `recipe-audit-formula-2.0`, each sequence
limit, the effective limit, and the recipe fingerprint used for every derived
group and device count.

This schema does not represent pipeline parallelism or separate prefill and
decode worker pools. Adapters must reject those topologies rather than map them
to TP/DP/EP defaults.

## Model resolution

`model-resolution-draft-1.0` is a config-level inspection result, not a model
manifest and not a planning input. It records the immutable repository revision,
discovered architecture and memory-related facts, field-level provenance,
warnings, and unresolved inputs. `ready=true` means the config and supplied
overrides contain the fields needed to proceed to SafeTensors inspection; it
does not mean the model fits any hardware.

`model-manifest-1.0` is the strict planning boundary. Creating it also requires
SafeTensors metadata and provenance. Packed tensor elements are never used as
logical model parameters. Mixed or unsupported weight layouts require measured
resident bytes. Custom, MLA, and hybrid cache capacity is deliberately absent
from the static model manifest and must come from the runtime inventory. A
draft cannot be passed to `explore` or `plan`.
Missing weight dtype is also unresolved; model resolution never defaults it to
BF16.

Online model resolution accepts a branch, tag, or commit SHA. A mutable
reference is resolved through repository metadata first. Config, SafeTensors,
cache keys, drafts, and manifests then use the returned 40-character commit
SHA. The model-info response must identify the requested repository when it
contains an ID, and its SHA must be valid. The offline manifest importer accepts
only an immutable SHA.

`model-planning-result-1.0` joins online resolution to single-model provider
planning without weakening the manifest boundary. `status=planned` contains a
complete `capacity-plan-1.0`. `status=needs-model-inputs` contains one
unresolved `model-resolution-draft-1.0` and no plan. Resolution errors other
than missing caller evidence remain errors. The planner never accepts the
draft itself.

## Provider planning

`provider-inventory-1.0` records caller-supplied instance shapes. Price and
availability require a timestamp. Prices also require a three-letter currency
code. The library does not fetch catalogs, convert currencies, or treat a
reported maximum as reserved capacity.

`runtime-inventory-1.0` records explicit runtime options, including engine and
version, TP/EP layout, memory reserves, KV capacity mode, llm-d limits, and
optionally the accelerator identities known to support the option. An empty
accelerator list is an unverified caller assumption, which appears as a warning
rather than a compatibility claim.

A resolved quantized model manifest cannot use nominal parameter-count-by-bit
width arithmetic as a resident-memory claim. A one-device candidate may use
measured total resident bytes from the manifest. A multi-device candidate must
use measured per-device resident bytes from the exact runtime layout. Mixed
quantization layouts require measured total resident bytes while the manifest
is being resolved.

`explore` and `plan` use the same candidate evaluator. Both run the capacity
calculator and the serving-recipe audit. Exploration reports per-replica and
per-instance sequence capacity at one context. Planning applies a load,
instance packing, price, and availability to each homogeneous candidate. One
TP replica must fit within one provider instance; the current solver does not
build a tensor-parallel replica across instances.

In a capacity plan, candidate `status` applies to the proposed resource claim.
`single_group_audit` applies the requested load to one generated serving group,
which is the baseline used to derive the resource claim. Keeping those scopes
separate avoids presenting a scalable plan and its one-group shortfall as
contradictory results.

A resource claim distinguishes:

- serving replicas and devices used by those replicas;
- whole provider instances and all devices allocated with them; and
- allocated devices left idle by TP packing or the final partially filled
  instance.

RPS, prefill TPS, decode TPS, TTFT, and TPOT still require an exact matching
measurement. `measurement-inventory-1.0` indexes measured group profiles by
recipe fingerprint, context, and request shape. A measurement for one provider
or runtime option is never copied to another candidate.

Lowest-cost ranking is deterministic only within one currency. Mixed-currency
inventories are rejected unless the caller chooses a non-cost objective or
normalizes the prices before calling the library.

When the caller supplies a `model-manifest-1.0`, exploration and planning keep
the full manifest at the result root and attach its evidence to each generated
recipe. Supplying a bare `ModelSpec` leaves `model_manifest` null.

## Partial recipe import

`recipe-draft-1.0` is a partial normalization boundary for existing llm-d
values. It preserves missing facts as structured `UnresolvedFact` records.
`structural-recipe-audit-1.0` may check TP/DP device assignment, configured
concurrency, runtime/llm-d block agreement, and flow-control shape before model
memory is known. It cannot report memory fit or traffic capacity.

A draft becomes a `serving-recipe-1.0` only after every required identity,
memory, topology, and routing field is populated. The materialization helper
requires a pinned model manifest and a matching measured initialization
profile. It rejects model identity or revision mismatches, configured context
above the manifest limit, mismatched host/runtime/topology identity, and a KV
capacity envelope whose hardware or KV memory budget does not reconcile with
the measured memory ledger.

Structural flow-control concurrency is block aligned when a block size is
known: complete flow-control blocks are divided by the blocks required for one
maximum-context sequence. Raw token division must not overstate a boundary
case.

## Model and vLLM evidence import

`model-manifest-1.0` records immutable model facts, serialized artifact bytes,
and field-level provenance. Serialized SafeTensors bytes are not resident GPU
bytes. SafeTensors shard headers, or the header from one `model.safetensors`
file, provide exact tensor shapes without requiring tensor payloads. Their
stored tensor element count is not treated as the model's logical parameter
count because packed quantization and scale tensors break that equivalence. A
quantized manifest needs the logical parameter count from the config or caller;
the resolver does not derive it from artifact size, stored shapes, or nominal
dtype. For a single-file artifact, the manifest identifies header data offsets
as the artifact-byte source instead of claiming SafeTensors index metadata.

When an unquantized config omits its parameter count, the online resolver may
use `safetensors.total` from model metadata whose returned commit matches the
requested immutable revision. The manifest records that exact source. The
fallback is forbidden for quantized models because packed tensors and scale
state break the equivalence.

The provider planner accepts either the manifest or its contained `ModelSpec`.
Passing the manifest avoids a manual translation step and preserves the same
model identity, quantization, and context. Model resolution
remains optional; cached manifests and caller-fetched metadata support fully
offline planning.

`vllm-initialization-profile-2.0` carries measured per-device resident weights,
runtime reserve, activation reserve, available KV memory, `max_num_seqs`, and
one or more context-bound sequence-capacity points. Its model revision must
match the manifest used to complete the draft. Hardware ID, runtime
engine/version, TP/DP/EP, physical device count, memory utilization, KV dtype,
block size, and runtime concurrency limit must also match the draft. One
measured evidence record binds the identity and memory metrics plus a canonical
digest of the capacity envelope.

The vLLM benchmark importer derives RPS, average request shape, prefill TPS, and
decode TPS from completed requests, duration, and total token counters. This
keeps all rates arithmetically consistent. It binds those fields and the exact
requested latency percentile to one recipe fingerprint in one measured
evidence record.

## Schema history

The schemas describe normalized output from `to_dict()`, including nullable and
defaulted fields. Hand-written CLI input documents may omit defaults and are
validated by the dependency-free Python parsers.

- `capacity-contract-1.0`: historical alpha schema with an invalid scalar
  sequence bound; retained for reference only.
- `capacity-contract-2.0`: historical schema with linear block/token budgets.
- `capacity-contract-3.0`: current schema with explicit linear and exact
  context-envelope capacity modes.
- `serving-recipe-1.0` and `load-requirement-1.0`: host topology and requested
  load.
- `recipe-audit-1.0`: historical audit without flow-aware group sizing.
- `recipe-audit-2.0`: current audit with explicit memory, runtime, llm-d flow,
  configured-concurrency, and measured limits.
- `recipe-draft-1.0` and `structural-recipe-audit-1.0`: partial import and
  topology/routing checks before model memory is known.
- `model-resolution-draft-1.0`: config-derived model facts, provenance, and
  unresolved inputs before weight metadata or runtime evidence is available.
- `model-manifest-1.0`: pinned static model metadata.
- `vllm-initialization-profile-1.0`: historical scalar KV profile.
- `vllm-initialization-profile-2.0`: current measured runtime memory and exact
  context-capacity profile.
- `provider-inventory-1.0`, `runtime-inventory-1.0`, and
  `measurement-inventory-1.0`: caller-supplied planning inputs.
- `capacity-exploration-1.0` and `capacity-plan-1.0`: ranked single-model
  candidate results and resource claims.
- `model-planning-result-1.0`: one-call public-model resolution and provider
  planning, with missing model evidence returned as data.
