# Roadmap and adoption gates

Keep the calculations and contracts in the standalone library. Services and
autoscaler adapters should call it instead of reimplementing its rules.

## Phase 0: correctness and OSS hygiene

- replace scalar concurrency with a KV/context envelope;
- make tensor-parallel KV layout explicit;
- return candidate-specific incompatibilities as data;
- ship a working installed `icc` command;
- enforce tests, Ruff, mypy, and schema validation in CI;
- tighten the 2.0 schemas; and
- align README and security guidance with repository behavior.

The 3.0 capacity contract now separates uniform linear KV arithmetic from
exact context-bound capacity for hybrid, MLA, custom, and sub-byte layouts. A
runtime envelope is reusable only when its hardware identity and KV memory
budget match.

This phase is done when source-tree and installed-wheel checks pass on Python
3.12 and 3.13.

## Phase 0.5: audit an existing deployment recipe

- normalize model, host, runtime, TP/DP/EP topology, and llm-d settings;
- calculate context-specific group capacity across independent KV ranks;
- check llm-d flow-control, concurrency, and block settings;
- accept direct RPS or TPS demand;
- bind measured traffic and latency to an exact recipe fingerprint and
  operating point; and
- report required groups, required devices, issues, and missing evidence.

This phase is done when sanitized TP and DP/EP recipes pass library, CLI,
schema, source-tree, and installed-wheel tests.

## Phase 1: model resolution and quantized artifacts

Building blocks now available: mutable-reference pinning, immutable revision
enforcement, optional metadata resolution, SafeTensors header-range parsing,
offline manifest caches, partial llm-d recipe import, vLLM initialization
evidence, and benchmark operating-point import. Nested text configs, explicit
head dimensions, hybrid attention, and mixed quantization are recognized;
unresolved model facts are returned in a versioned model-resolution draft,
while runtime-specific KV capacity stays in the initialization profile and
runtime inventory.

- [x] resolve Hugging Face references to immutable revisions;
- parse config and SafeTensors metadata without downloading full weights;
- cache normalized manifests for offline use;
- calculate actual resident bytes for real quantized artifacts; and
- attach field-level provenance.

This phase is done when one public model and one real quantized artifact resolve
to pinned manifests that can be replayed offline.

## Phase 2: single-model analytical solver

Building blocks now available: caller-supplied provider and runtime
inventories, shared `explore` and `plan` evaluation, whole-instance resource
claims, price and availability snapshots, candidate rejection reasons, and
flow-aware recipe audits. Both planner APIs accept cached model manifests
directly, and the CLI can optionally resolve a pinned public model before
planning offline.

- accept normalized user-supplied provider inventories;
- add a versioned vLLM runtime adapter;
- expose `explore` and `plan` over the same candidate evaluator;
- [x] add a one-call public-model-to-provider planning workflow;
- return ranked plans, rejected-candidate reasons, uncertainty, and resource
  claims;
- reuse the recipe auditor for every proposed candidate;
- preserve model-manifest evidence in every generated plan; and
- property-test monotonicity and forward/reverse consistency.

This phase is done when one resolved model can be checked against multiple
providers without a measured performance profile. Every result must say whether
it is analytical or estimated.

## Phase 3: measured SLO planning

- define operating-point performance profiles;
- import `vllm bench serve` results;
- match exact model/runtime/hardware/quantization variants;
- filter by RPS/TPS/TTFT/TPOT/E2E constraints; and
- publish prediction-error and initialization-validation tables.

This phase is done when the planner can solve the same measured workload from
either direction and return consistent capacity and SLO results.

## Later phases

- SGLang runtime adapter;
- live provider catalog importers;
- multi-model `plan_portfolio` placement and allocation;
- optional HTTP service; and
- shadow/autoscaler integrations after recommendation error is known.

## Stability gate

Do not call the schema stable until at least two independent consumers use it,
NVIDIA and AMD paths have real validation runs, and the project publishes its
prediction errors and failure cases.
