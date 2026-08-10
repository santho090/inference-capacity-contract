# Changelog

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
