# Capture a vLLM initialization profile

ICC needs the byte values and group-aware KV capacity from the same successful
vLLM initialization. Do not reconstruct them from rounded GiB log lines.

For the pinned vLLM build under test, export these values from each worker:

| ICC field | vLLM value |
| --- | --- |
| `device_memory_bytes` | `init_snapshot.total_memory` |
| `weight_bytes_per_device` | `model_runner.model_memory_usage` |
| `activation_reserve_bytes_per_device` | `peak_activation_memory` |
| `kv_capacity_memory_bytes_per_device` | `available_kv_cache_memory_bytes` |
| `runtime_overhead_bytes_per_device` | `requested_memory - weight_bytes_per_device - activation_reserve_bytes_per_device - kv_capacity_memory_bytes_per_device` |

All memory fields are bytes. The four budget components must add up to that
worker's requested memory. The adapter also checks the requested memory against
`ceil(device_memory_bytes * memory_utilization_limit)`. It accepts different
component ledgers and uses the worker with the least available KV memory as the
limiting worker. It rejects workers with different device sizes or requested
budgets because ICC's hardware model has one memory budget per device.

The rest of the profile must come from the exact run configuration: model
revision, hardware ID, runtime image or commit, TP/DP/EP sizes, physical device
count, memory-utilization limit, KV dtype, block size, and `max_num_seqs`.

For uniform full-attention models, ICC can calculate sequence capacity from
the memory ledger. Hybrid, MLA, custom, and sub-byte caches need vLLM's
group-aware result. Capture the active context and the maximum concurrency
returned by `get_kv_cache_capacity` after vLLM builds the final
`KVCacheConfig`. Record both returned values as `kv_cache_size_tokens` and
`kv_cache_max_concurrency`. The adapter checks their arithmetic and stores the
whole-number safe capacity as:

```json
{
  "context_tokens": 1048576,
  "max_sequences": 24
}
```

The adapter rounds fractional concurrency down. One raw snapshot represents
one exact context. Keep separate snapshots for other contexts; ICC deliberately
refuses to interpolate between them.

The current vLLM source used to define this adapter boundary is pinned at
`f7ef489e93cf92b8d6ce7403b49f1db867bcc35e`. See its
[`MemoryProfilingResult`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/utils/mem_utils.py),
[`GPUWorker.determine_available_memory`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/worker/gpu_worker.py),
and
[`get_kv_cache_capacity`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/core/kv_cache_utils.py).

Convert the raw worker and scheduler snapshot with:

```bash
icc import-vllm-snapshot \
  --input vllm-runtime-snapshot.json \
  --source benchmark://initialization-run \
  --output initialization-profile.json
```

[`vllm-runtime-snapshot.json`](fixtures/vllm-runtime-snapshot.json) shows the
complete input. `icc import-vllm-init` remains available when another adapter
already emits a normalized initialization profile.

The adapter binds the full identity, limiting memory ledger, `max_num_seqs`,
raw worker count, group-aware scheduler values, and snapshot digest into one
measured evidence record. A changed hardware ID, runtime version, topology,
memory budget, or envelope invalidates reuse.
