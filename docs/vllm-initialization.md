# Capture a vLLM initialization profile

ICC needs the byte values and group-aware KV capacity from the same successful
vLLM initialization. Do not reconstruct them from rounded GiB log lines.

For the pinned vLLM build under test, export these values from each worker:

| ICC field | vLLM value |
| --- | --- |
| `weight_bytes_per_device` | `model_runner.model_memory_usage` |
| `activation_reserve_bytes_per_device` | `peak_activation_memory` |
| `kv_capacity_memory_bytes_per_device` | `available_kv_cache_memory_bytes` |
| `runtime_overhead_bytes_per_device` | `requested_memory - weight_bytes_per_device - activation_reserve_bytes_per_device - kv_capacity_memory_bytes_per_device` |

All four fields are bytes. The sum must equal vLLM's requested per-device
memory budget. Initialization profile 2.0 assumes the ranks have the same
ledger. If they differ, keep the run unresolved; do not combine maxima and
minima from different ranks into a ledger that no worker had.

The rest of the profile must come from the exact run configuration: model
revision, hardware ID, runtime image or commit, TP/DP/EP sizes, physical device
count, memory-utilization limit, KV dtype, block size, and `max_num_seqs`.

For uniform full-attention models, ICC can calculate sequence capacity from
the memory ledger. Hybrid, MLA, custom, and sub-byte caches need vLLM's
group-aware result. Capture the active context and the maximum concurrency
returned by `get_kv_cache_capacity` after vLLM builds the final
`KVCacheConfig`. Store the whole-number safe capacity as:

```json
{
  "context_tokens": 1048576,
  "max_sequences": 24
}
```

If the runtime reports a fractional concurrency, round down. To support more
than one context, initialize or measure each context with the same recipe and
add one point per run. ICC deliberately refuses to interpolate between points.

The current vLLM source used to define this adapter boundary is pinned at
`f7ef489e93cf92b8d6ce7403b49f1db867bcc35e`. See its
[`MemoryProfilingResult`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/utils/mem_utils.py),
[`GPUWorker.determine_available_memory`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/worker/gpu_worker.py),
and
[`get_kv_cache_capacity`](https://github.com/vllm-project/vllm/blob/f7ef489e93cf92b8d6ce7403b49f1db867bcc35e/vllm/v1/core/kv_cache_utils.py).

Normalize the captured JSON with:

```bash
icc import-vllm-init \
  --input vllm-initialization.json \
  --source benchmark://initialization-run \
  --output initialization-profile.json
```

The importer binds the full identity, memory ledger, `max_num_seqs`, and a
digest of every capacity point into one measured evidence record. A changed
hardware ID, runtime version, topology, memory budget, or envelope invalidates
reuse.
