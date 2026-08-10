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
- llm-d planner and scaling-policy payloads; and
- measured-profile replica recommendations with cost and GPU-hour deltas.

The library does not start vLLM or SGLang, discover GPUs, mutate a cluster, or
claim throughput and latency from model parameters. Runtime initialization and
SLO performance require matching measured evidence.

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
python -m pip install .
icc --help
```

The runtime package has no third-party dependencies. Development tools are
installed separately with `python -m pip install -e '.[dev]'`.

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

icc export \
  --contract contract.json \
  --target llmd-planner
```

`fit` checks every hardware and runtime pair. If a pair is unsupported, the
result includes `unsupported_reason` and the search continues.

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

These schemas describe normalized output documents produced by `to_dict()`.
CLI input files may omit nullable/defaulted fields; the dependency-free Python
parsers enforce their input contracts directly.

## Roadmap

The next milestone expands the library into a single-model planner:

1. resolve pinned Hugging Face artifacts and exact tensor bytes;
2. accept normalized user-supplied provider inventories;
3. expose `explore` for supply-to-capacity and `plan` for demand-to-supply;
4. import measured vLLM profiles for RPS/TPS/TTFT/TPOT planning; and
5. later add SGLang and a separate multi-model portfolio planner.

No HTTP service or autoscaler integration precedes a validated library contract.
See `docs/roadmap.md`.

## Project status

Alpha. Treat analytical fit as a filter, not a deployment guarantee. Stability
requires real NVIDIA and AMD initialization runs, measured profile imports, and
a published prediction-error/failure table.

## License

Apache-2.0. See `LICENSE`.
