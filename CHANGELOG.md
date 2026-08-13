# Changelog

## 0.5.0

- Makes serving-recipe group sizing honor the block-aligned llm-d flow-control
  budget at the requested context.
- Reports every sequence-capacity limit and the effective per-group limit.
- Returns no finite group count when a group cannot serve one requested
  sequence.
- Adds the `recipe-audit-2.0` schema. The historical 1.0 schema remains for
  reference.
- Adds caller-supplied provider, runtime, and measurement inventories.
- Adds `explore` and `plan` APIs plus `icc explore` and `icc plan-providers`.
- Reports whole-instance allocation, serving and idle devices, cost currency,
  point-in-time availability, and candidate-specific rejection reasons.
- Refuses lowest-cost ranking when supplied prices use different currencies.
- Lets `explore` and `plan` consume replayable model manifests directly.
- Adds `icc resolve-model` for optional pinned public-model metadata resolution
  with offline cache replay.
- Lets online model inspection and resolution accept a branch or tag, default
  to `main`, and pin it before artifact access or cache lookup.
- Adds `plan_huggingface_model` and `icc plan-model` to turn a public model ID,
  caller inventories, and load into either a ranked plan or typed missing
  model inputs.
- Adds `icc inspect-model` and `model-resolution-draft-1.0` so discovered model
  facts and unresolved manifest inputs are returned as data.
- Uses revision-bound SafeTensors parameter totals when an unquantized config
  omits its logical parameter count.
- Resolves both sharded SafeTensors indexes and single-file
  `model.safetensors` artifacts by reading headers only.
- Keeps a supplied model manifest and its evidence in exploration and planning
  results.
- Names the nested candidate check `single_group_audit` so it cannot be
  confused with the proposed multi-group plan.
- Carries non-aligned llm-d flow-control warnings into provider plans.
- Records single-file artifact bytes as SafeTensors header-offset evidence,
  rather than index metadata.

## 0.4.0

- Adds dependency-free import of caller-parsed llm-d values into partial
  recipe drafts.
- Adds structural audits for TP/DP device assignment, configured concurrency,
  KV block agreement, and flow-control shape before model memory is known.
- Resolves only immutable Hugging Face revisions and caches replayable model
  manifests.
- Records stored tensor element counts from SafeTensors header ranges without
  downloading tensor payloads.
- Keeps serialized artifact bytes separate from measured resident GPU bytes.
- Imports vLLM initialization memory facts and exact benchmark operating points.
- Materializes a complete recipe only when model, runtime, hardware, topology,
  context, and measured KV capacity agree.
- Makes structural flow-control concurrency block aligned.
- Prevents serialized drafts from omitting required unresolved facts.
- Preserves matching immutable model revisions found in llm-d metadata,
  artifact URIs, or runtime flags and rejects conflicting revisions.
- Adds sanitized long-context TP8 and DP4-on-eight-device golden cases.

Provider discovery, performance prediction, and live infrastructure changes
remain outside the library. Traffic sizing still requires measurements from
the exact recipe fingerprint and operating point.

## 0.3.0

- Adds a normalized serving recipe for TP, DP, EP, physical device, and
  independent KV-rank layouts.
- Adds `audit_recipe` and `icc audit` to check a configured vLLM or SGLang
  shape against context, concurrency, measured traffic, and latency targets.
- Checks llm-d flow-control, concurrency, and KV block settings against the
  calculated runtime capacity.
- Accepts RPS or direct prefill/decode TPS demand.
- Binds each measured metric to an exact serving-recipe fingerprint and
  records its context and request shape.
- Versions the recipe fingerprint and requires matching latency percentiles.
- Reports whether the recipe is sufficient, insufficient, incomplete, or
  invalid, along with required groups and devices.
- Adds public schemas and sanitized fixtures for recipe and load audits.

Traffic and latency planning still requires a measured group profile from the
exact model, runtime, hardware, topology, and routing combination. The audit
does not estimate throughput from parameter count.

## 0.2.0

- Replaces the invalid context-free sequence scalar with a context-dependent
  concurrency envelope.
- Separates per-device KV bytes, blocks, and tokens.
- Treats a runtime variant as one tensor-parallel serving replica.
- Requires explicit TP KV layouts and runtime-specific overrides for hybrid,
  MLA, sub-byte, and custom cache layouts.
- Adds strict semantic validation, analytical provenance, measured-concurrency
  scaling, GPU-hour estimates, and source/wheel build checks.

This release introduces the breaking `capacity-contract-2.0` and
`scaling-recommendation-2.0` schemas. Historical 1.0 schemas remain under
`schemas/` for reference only.
