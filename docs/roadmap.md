# Roadmap and adoption gates

The launch wedge is a portable capacity contract, not a universal inference control plane.

## Phase 0: public core (current)

Ship the dependency-free contract, deterministic memory/KV calculation, reverse-fit query, JSON schema, fixtures, and non-actuating planner/scaling exports.

Exit criteria:

- source tests and installed-wheel smoke tests pass;
- NVIDIA and AMD fixtures exercise both supported vendors;
- unsupported topology, unsupported vendor, unknown context, and custom-attention cases are explicit;
- every derived value carries assumptions or warnings; and
- no private catalog, credential, cluster client, or deployment mutation exists in the default package.

## Phase 1: measured profile import (experimental implementation included)

The current package includes a first rate-based `WorkloadProfile` and `scaling-recommendation-1.0` calculator. Extend it with a versioned workload document containing request-rate, prompt/output distributions, concurrency, queue, TTFT/TPOT/E2E percentiles, KV occupancy, readiness time, and failure boundaries. Import benchmark results as `EvidenceRecord` values without changing the analytical calculator.

Exit criteria:

- a profile is reproducible from a pinned model revision, runtime version, hardware topology, and workload seed;
- analytical prediction error is reported rather than hidden; and
- initialization and SLO validation levels are granted only by explicit validation workflows.

## Phase 2: upstream adapter

Implement one adapter against a pinned llm-d planner contract. Keep the core schema independent from upstream release churn and test the translation in an integration fixture. Add a second consumer adapter for a workload-aware WVA/KEDA/HPA policy input; it may calculate replicas only when measured throughput/latency evidence is present.

Exit criteria:

- one producer and two consumer adapters exist;
- warnings and validation levels survive every translation;
- no adapter emits deployment actions as a side effect; and
- upstream compatibility is tested against a pinned version.

## Phase 3: cost and fleet extensions

Add private, separately versioned adapters for GPU discovery, price catalogs, fleet availability, and organization policy. Keep these out of the public core. Cost-saving suggestions should be expressed as ranked, evidence-backed alternatives with a stated confidence interval and a rollback path.

## Standalone-repository gate

Treat the schema as stable only after at least two independent consumers use it, NVIDIA and AMD paths have validation runs, and the project publishes prediction-error and failure cases. Until then, breaking schema changes are acceptable when they remove ambiguity or false precision.

## Explicit non-goals

Do not add live autoscaling, queue admission, deployment generation, GPU probing, or model downloading until Phase 1 establishes the workload evidence required to make those actions falsifiable.
