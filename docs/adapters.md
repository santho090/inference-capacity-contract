# Adapter boundary

The core returns facts. Adapters translate those facts into the vocabulary of a planner or autoscaler while preserving uncertainty.

## llm-d planner

`to_llmd_planner_payload` emits `llmd-capacity-input-1.0`. It includes the immutable model revision, hardware topology, runtime variant, memory breakdown, KV bounds, validation level, warnings, and evidence. It does not claim to be an llm-d-native API; an integration should map this payload to the pinned planner version and test that mapping against the upstream contract.

## Workload-aware scaling

`to_scaling_policy_input` emits `scaling-policy-input-1.0`. It provides per-replica token and sequence bounds but intentionally leaves `replica_count` null. A WVA/KEDA/HPA adapter must combine these bounds with observed request rate, prompt/output distributions, queueing, and SLO measurements before calculating replicas.

`recommend_scale` is the library's transparent rate-based calculation for that measured-profile step. Its `scaling-recommendation-1.0` output can be consumed by a policy adapter, but it never writes a deployment or chooses a cloud fleet.

## What an adapter must not do

- turn `fits=true` into a production guarantee;
- treat analytical KV capacity as throughput;
- hide warnings or downgrade evidence provenance;
- invent prices, fleet availability, or a target replica count; or
- mutate a cluster as a side effect of serialization.
