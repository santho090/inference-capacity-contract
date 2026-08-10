# Capacity contract specification

Current schema identifier: `capacity-contract-2.0`

The JSON document is the durable interchange format. Python dataclasses are a
convenience layer over the same fields.

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

`concurrency_envelope` materializes this function at useful context lengths.
Consumers may call the Python contract's `max_sequences_at(context_tokens)` for
another point. The schema deliberately has no context-free maximum sequence
field because the quantity is not a scalar.

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
Peak concurrency additionally requires `concurrency_context_tokens` and a
measured `sustainable_concurrent_sequences_per_replica`. The solver uses the
lower of that measurement and the contract's context-specific analytical KV
bound.

The recommendation reports replica count, GPU-hours per hour, GPU-hour delta,
and—when price is available—hourly cost and cost delta. It is a policy input,
not an actuation command.

## Schema history

- `capacity-contract-1.0`: historical alpha schema with an invalid scalar
  sequence bound; retained for reference only.
- `capacity-contract-2.0`: current breaking schema with block/token budgets and
  a context-dependent concurrency envelope.
