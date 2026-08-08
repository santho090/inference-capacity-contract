# Project context

`inference-capacity-contract` is a public, dependency-free contract core for LLM serving capacity facts.

## Vocabulary

- **Model artifact**: model identity plus immutable revision and architecture/weight facts.
- **Runtime variant**: engine, version, parallelism, cache dtype, and explicit reserves.
- **Hardware topology**: vendor, device count, memory, and optional price/interconnect metadata.
- **Evidence record**: provenance for analytical, measured, reported, or extrapolated claims.
- **Capacity contract**: versioned descriptive output; never an actuation command.
- **Scaling policy input**: per-replica facts consumed by a separate workload-aware autoscaling adapter.

## Design rule

Static memory fit, runtime initialization, SLO performance, and live autoscaling are different claims. The public core must keep them separate and preserve uncertainty in machine-readable output.
