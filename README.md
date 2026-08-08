# Inference Capacity Contract

`inference-capacity-contract` is a small, dependency-free Python library for answering one narrow question:

> Given a model artifact, a pinned inference runtime, and a hardware topology, what can we prove about one serving replica before we run it?

It produces a versioned JSON contract containing:

- weight, runtime-reserve, activation-reserve, and KV-cache memory accounting;
- model-to-hardware compatibility checks, including vendor and parallelism topology;
- KV bytes per token, token capacity, context bound, and sequence bound;
- explicit assumptions, warnings, validation level, and evidence provenance;
- reverse-fit results across a hardware inventory; and
- neutral exports for an llm-d planner adapter or a workload-aware scaling adapter.
- measured-profile replica recommendations with cost and GPU-hour deltas.

The library is intentionally descriptive. It does not start vLLM or SGLang, discover GPUs, emit deployment manifests, set HPA/KEDA replicas, or claim TTFT/TPOT/throughput. Those require runtime initialization and measured workload evidence.

## Why this exists

Serving systems often mix four different claims: static memory fit, runtime initialization success, SLO performance at a workload, and a live autoscaling policy. This project keeps those claims separate. A deterministic analytical result is useful as a contract input, but it is not a production capacity guarantee.

## Quick start

Requires Python 3.12+ and has no runtime dependencies.

```python
from inference_capacity_contract import HardwareSpec, ModelSpec, RuntimeVariant, capacity_for

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
)

contract = capacity_for(model, hardware, runtime)
print(contract.to_dict())
```

The same calculation is available through `icc plan`, `icc fit`, `icc validate`, `icc export`, and `icc scale`. JSON fixtures are in [`docs/fixtures`](docs/fixtures).

## Calculation boundary

For the standard attention path, KV bytes per token are calculated as:

```text
2 × num_layers × num_kv_heads × head_dim × bytes_per_KV_element
```

Usable memory is `device_memory × memory_utilization_limit`. The contract subtracts per-device weights, runtime overhead, and activation reserve, then divides the remainder by KV bytes per token. Weights can be supplied explicitly for quantized, sharded, or otherwise non-ideal artifacts.

MLA and custom attention require an explicit `kv_bytes_per_token_override`; the library will not guess.

## Validation levels

`analytically-feasible` is the only level produced by `capacity_for` in v0. The public enum reserves a monotonic vocabulary for adapters and future validation workflows:

- `runtime-supported`: the pinned runtime declares support;
- `initialization-validated`: the exact variant initialized successfully; and
- `slo-validated`: a workload benchmark established a stated SLO envelope.

Attaching an [`EvidenceRecord`](src/inference_capacity_contract/models.py) never silently promotes a contract. A producer must run and record the validation step that justifies a higher level.

## Evidence-backed scaling

`WorkloadProfile` and `recommend_scale` provide the first useful autoscale calculation without turning static memory fit into a throughput claim. A profile binds to an exact model revision, hardware id, and runtime version; it requires measured evidence and one or more sustainable per-replica rates. The recommendation takes the maximum replica requirement across request rate, prefill tokens, decode tokens, and optional peak concurrency, then applies utilization and headroom buffers. If hardware pricing and a baseline replica count are present, it reports estimated hourly cost and GPU-hour savings.

The output is a policy input, not a cluster mutation. Missing measured evidence, scope mismatches, unknown prices, and bounds that cannot satisfy demand remain explicit warnings.

## Non-goals for v0

- no live GPU probing or cluster mutation;
- no throughput, TTFT, TPOT, queueing, or cost forecast;
- no desired replica count and no autoscaling decision;
- no deployment YAML generator;
- no private cloud SKU catalog, credentials, or internal measurements.

## Integrations

`to_llmd_planner_payload(contract)` exports model/runtime/hardware facts and feasibility bounds for an llm-d planner adapter. `to_scaling_policy_input(contract)` exports per-replica bounds while setting `replica_count` to `null` and requiring an observed workload profile. These are deliberately small translation seams, not upstream API claims.

See [`docs/specification.md`](docs/specification.md), [`docs/adapters.md`](docs/adapters.md), and [`docs/publication-policy.md`](docs/publication-policy.md).

## Project status

Alpha. The standalone-repository gate is intentionally empirical: adoption should be demonstrated by at least two independent consumers, one producer adapter, two consumer adapters, NVIDIA and AMD validation, and published prediction-error/failure cases before the contract is treated as stable.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
