# Capacity contract specification

Current schema identifier: `capacity-contract-2.0`

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
| `runtime` | engine/version, TP size, KV dtype, block size, reserves | One pinned runtime variant. TP greater than one requires an explicit per-device KV-head layout; exact non-uniform weight layouts use a per-device byte override. |

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

For uniform full self-attention with equal K/V dimensions, the calculator
derives per-device KV bytes/token from the layers, KV heads resident on that
device, head dimension, and KV dtype. MLA, hybrid cache groups, unequal K/V
dimensions, sub-byte formats, and custom attention require an explicit
per-device override.

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

`concurrency_envelope` records this function at useful context lengths.
Consumers may call the Python contract's `max_sequences_at(context_tokens)` for
another point. The schema has no context-free maximum sequence field because
the quantity is not a scalar.

These capacities are per device because every TP device stores a shard of the
same sequences. They are already the replica bottleneck and must not be
multiplied by `hardware.device_count`.

## Compatibility and failure behavior

`fits=true` requires:

1. declared runtime support for the hardware vendor;
2. hardware device count equal to tensor-parallel size;
3. a known per-device KV layout for TP greater than one; and
4. enough usable memory for weights, reserves, and at least one KV block.

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
lower of that measurement and the contract's context-specific analytical KV
bound.

The recommendation reports replica count, GPU-hours per hour, and GPU-hour
delta. When a price is available, it also reports hourly cost and cost delta.
The result is a policy input, not an actuation command.

## Serving recipe audit

`serving-recipe-1.0` adds the host-level topology that a per-replica contract
does not contain. `tensor_parallel_size * data_parallel_size` is the number of
engine ranks. `physical_device_count` is the number of devices allocated to
the group. `independent_kv_ranks` says how many ranks own separate KV pools.

The audit calculates one KV-rank contract using the TP device count, then
multiplies sequence and KV-token capacity by the number of independent KV
ranks. It does not multiply capacity by expert-parallel size. EP changes model
placement, so an EP recipe must supply resident weight bytes per device.

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
concurrent sequences with the calculated runtime values. A mismatch makes the
recipe `invalid`. The result records `recipe-audit-formula-1.0` and the recipe
fingerprint used for every derived group and device count.

This schema does not represent pipeline parallelism or separate prefill and
decode worker pools. Adapters must reject those topologies rather than map them
to TP/DP/EP defaults.

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
capacity that does not reconcile with the measured memory ledger.

Structural flow-control concurrency is block aligned when a block size is
known: complete flow-control blocks are divided by the blocks required for one
maximum-context sequence. Raw token division must not overstate a boundary
case.

## Model and vLLM evidence import

`model-manifest-1.0` records immutable model facts, serialized artifact bytes,
and field-level provenance. Serialized SafeTensors bytes are not resident GPU
bytes. SafeTensors shard headers provide exact tensor shapes without requiring
the tensor payloads. Their stored tensor element count is not treated as the
model's logical parameter count because packed quantization and scale tensors
break that equivalence. A quantized manifest needs the logical parameter count
from the config or caller; the resolver does not derive it from artifact size,
stored shapes, or nominal dtype.

`vllm-initialization-profile-1.0` carries measured per-device resident weights,
runtime reserve, activation reserve, KV bytes/token, and KV token capacity. Its
model revision must match the manifest used to complete the draft. Hardware ID,
runtime engine/version, TP/DP/EP, physical device count, memory utilization, KV
dtype, and block size must also match the draft.

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
- `capacity-contract-2.0`: current breaking schema with block/token budgets and
  a context-dependent concurrency envelope.
- `serving-recipe-1.0`, `load-requirement-1.0`, and `recipe-audit-1.0`: host
  topology, requested load, and the resulting configuration audit.
- `recipe-draft-1.0` and `structural-recipe-audit-1.0`: partial import and
  topology/routing checks before model memory is known.
- `model-manifest-1.0` and `vllm-initialization-profile-1.0`: pinned static
  metadata and measured runtime memory facts.
