# Project context

`inference-capacity-contract` is a public, dependency-free library for
calculating LLM serving capacity.

## Vocabulary

- **Model artifact**: model identity plus immutable revision and architecture/weight facts.
- **Runtime variant**: engine, version, parallelism, cache dtype, and explicit reserves.
- **Hardware topology**: vendor, device count, memory, and optional price/interconnect metadata.
- **Evidence record**: provenance for analytical, measured, reported, or extrapolated claims.
- **Capacity contract**: versioned capacity output that does not change infrastructure.
- **Recipe draft**: partial imported configuration with explicit unresolved facts.
- **Model manifest**: replayable facts from one immutable model artifact revision.
- **Initialization profile**: measured per-device runtime memory and KV facts.
- **Scaling policy input**: per-replica facts consumed by a separate workload-aware autoscaling adapter.

## Design rule

Static memory fit, runtime initialization, SLO performance, and live
autoscaling are different claims. The library reports them separately and
records uncertainty in its output.
