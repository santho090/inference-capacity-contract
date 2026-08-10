# Changelog

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
