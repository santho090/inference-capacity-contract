# Adapter boundary

The core returns facts. Adapters translate those facts into a planner or policy
vocabulary while preserving uncertainty. They never mutate infrastructure.

## llm-d planner payload

`to_llmd_planner_payload` emits `llmd-capacity-input-2.0`. It includes the
immutable model revision, hardware topology, runtime variant, per-device memory
ledger, KV block/token budget, context/concurrency envelope, validation level,
warnings, and evidence.

This is a neutral adapter payload, not a claim that the shape is an upstream
llm-d API. An integration must map it to a pinned llm-d version and test that
translation against the upstream contract.

## Scaling-policy payload

`to_scaling_policy_input` emits `scaling-policy-input-2.0`. It includes
per-replica KV and context/concurrency bounds while leaving `replica_count`
null. A consumer must combine it with measured workload evidence.

`recommend_scale` emits `scaling-recommendation-2.0` when an exact measured
profile is present. It reports replicas, cost, and GPU-hour deltas but never
writes a deployment or chooses a cloud fleet.

## Adapter rules

An adapter must not:

- turn `fits=true` into an initialization or production guarantee;
- treat analytical KV capacity as throughput;
- hide warnings or evidence provenance;
- invent a price, availability, performance profile, or target replica count;
- collapse a concurrency envelope into one context-free scalar; or
- mutate a cluster as a serialization side effect.
