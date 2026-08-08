# Capacity contract specification

Schema identifier: `capacity-contract-1.0`

The JSON form is the durable interchange format. Python dataclasses are a convenience layer over the same fields.

## Inputs

| Object | Required facts | Meaning |
| --- | --- | --- |
| `model` | id, immutable revision, parameter count, layer count, KV-head count, head dimension | Architecture and artifact facts. `explicit_weight_bytes` is preferred when the raw parameter count does not describe resident weights. |
| `hardware` | id, vendor, device count, memory per device | One replica topology, not a whole cluster. |
| `runtime` | engine, version, tensor/data parallelism, KV dtype | Pinned serving variant. Runtime and activation reserves are explicit inputs. |

## Output semantics

`fits` is true only when vendor support, device topology, and non-negative memory remainder all hold. It does not mean initialization or SLO success.

`memory_bytes_per_device` contains:

- `usable_budget`: device memory multiplied by the utilization limit;
- `weights`: resident weight bytes after tensor-parallel division (rounded up);
- `runtime_reserve`: declared runtime reserve;
- `activation_reserve`: declared activation reserve; and
- `kv_available`: the remainder available to KV storage.

`remaining_bytes_per_device` equals the non-negative KV remainder. It is not free device memory after a runtime has initialized.

`kv_capacity_tokens` is the integer floor of `kv_available / kv_bytes_per_token`. `max_context_tokens` is additionally capped by `model.max_model_len` when that value is known. `max_concurrent_sequences` is a conservative token-bound divided by runtime block size and capped by `max_num_seqs`, then multiplied by data parallelism.

`warnings` are actionable uncertainty or incompatibility notices. Consumers should surface them rather than discard them.

`evidence` records provenance; each record has an id, kind, source, scope, optional metrics, and optional collection timestamp.

## Scaling recommendation semantics

`scaling-recommendation-1.0` is derived from a capacity contract plus a measured `WorkloadProfile`. The profile must match model revision, hardware id, runtime engine, and runtime version exactly. For each available driver, required replicas are `ceil(demand / (sustainable_per_replica_rate × target_utilization))`. The recommendation is `max(min_replicas, ceil(max_driver × scale_up_buffer))`, capped by `max_replicas` when configured. `within_bounds=false` means the configured maximum cannot satisfy the measured demand.

The calculation is intentionally rate-based. It does not model queueing, burst distributions, cold-start time, or SLO tails; those belong in the measured profile and a later runtime-specific policy adapter.

## Compatibility rules

The v0 calculator rejects a candidate when:

1. the hardware vendor is not in the runtime's declared `supported_vendors`;
2. hardware device count differs from `tensor_parallel_size × data_parallel_size`; or
3. weights plus declared reserves exceed the usable memory budget.

The calculator does not infer support from a model name, GPU marketing name, or an unpinned runtime version.
