# Roadmap and adoption gates

The project is a library-first capacity solver. A service or autoscaler adapter
is downstream of a validated standalone library.

## Phase 0: correctness and OSS hygiene

- replace scalar concurrency with a KV/context envelope;
- make tensor-parallel KV layout explicit;
- return candidate-specific incompatibilities as data;
- ship a working installed `icc` command;
- enforce tests, Ruff, mypy, and schema validation in CI;
- tighten the 2.0 schemas; and
- align README and security guidance with repository behavior.

Exit gate: clean source and installed-wheel validation passes on Python 3.12
and 3.13.

## Phase 1: model resolution and quantized artifacts

- resolve Hugging Face references to immutable revisions;
- parse config and SafeTensors metadata without downloading full weights;
- cache normalized manifests for offline use;
- calculate actual resident bytes for real quantized artifacts; and
- attach field-level provenance.

Exit gate: one public model and at least one real quantized artifact resolve
reproducibly and replay offline.

## Phase 2: single-model analytical solver

- accept normalized user-supplied provider inventories;
- add a versioned vLLM runtime adapter;
- expose `explore` and `plan` over the same candidate evaluator;
- return ranked plans, rejected-candidate reasons, uncertainty, and resource
  claims; and
- property-test monotonicity and forward/reverse consistency.

Exit gate: one resolved model evaluates multiple providers without requiring a
measured performance profile and clearly labels analytical/estimated results.

## Phase 3: measured SLO planning

- define operating-point performance profiles;
- import `vllm bench serve` results;
- match exact model/runtime/hardware/quantization variants;
- filter by RPS/TPS/TTFT/TPOT/E2E constraints; and
- publish prediction-error and initialization-validation tables.

Exit gate: a measured demand plan can be explored in reverse with consistent
capacity and SLO results.

## Later

- SGLang runtime adapter;
- live provider catalog importers;
- multi-model `plan_portfolio` placement and allocation;
- optional HTTP service; and
- shadow/autoscaler integrations after recommendation error is known.

## Stability gate

Do not call the schema stable until at least two independent consumers use it,
NVIDIA and AMD paths have real validation runs, and the project publishes its
prediction errors and failure cases.
