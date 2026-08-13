# Deployment-shaped audit cases

These fixtures preserve useful serving patterns without publishing private
model names, host identifiers, repository paths, prices, or benchmark results.
They are configuration tests, not performance claims.

## Long-context TP8 group

`llmd-values-long-context-tp8.json` describes an eight-device TP8/EP group with:

- a 1,048,576-token configured context;
- `max_num_seqs=24`;
- a 1,536-token KV block;
- a 2,104,030-token flow-control limit; and
- KV events enabled without an asserted routing policy.

The structural audit reports eight engine ranks, no unassigned devices, and a
block-aligned flow budget equal to two maximum-context sequences. It also warns
that the flow limit is not divisible by the KV block size. The 24-sequence
runtime limit therefore applies only below roughly 88K active tokens per
sequence; flow control permits two sequences at the configured 1M-token limit.
This is a routing-budget result, not proof of a memory overrun: runtime KV
capacity still requires an initialization measurement.

The complete sanitized inputs can also be sent through the provider planner:

```bash
icc plan-providers \
  --manifest docs/fixtures/model-manifest-long-context.json \
  --providers docs/fixtures/provider-inventory-long-context.json \
  --runtimes docs/fixtures/runtime-inventory-long-context.json \
  --load docs/fixtures/load-context-concurrency.json
```

For a peak of 30 full-context sequences at 80% target utilization, the
sanitized initialization envelope reports 24 sequences at the requested
context, but llm-d flow control allows only two. The binding limit therefore
requires 19 TP8 groups, or 152 serving devices. Price and availability remain
unknown because the fixture does not invent them.

### Applying the shape to Kimi K3

The pinned public Kimi K3 config can now be read without flattening its nested
text model into the wrong architecture. ICC identifies 93 layers, 96 KV heads,
an explicit value-head width of 128, a 1,048,576-token limit, hybrid attention,
and a mixed MXFP4 checkpoint layout.

That metadata does not make this sanitized TP8 fixture a measured Kimi K3
recipe. A real run still needs model and runtime evidence:

- the logical parameter count, because packed SafeTensors elements are not
  logical model parameters;
- resident weight bytes from the exact vLLM initialization and TP/EP layout;
  and
- context-bound KV sequence capacity from the same hardware, memory budget,
  and runtime configuration.

`icc inspect-model` returns the discovered fields and missing inputs as data.
Until those values are supplied, strict `icc resolve-model` still refuses to
write a manifest. Once they are supplied, the same planner can test the real
recipe's memory, runtime concurrency, and llm-d flow limits. Throughput and
latency sizing still needs a benchmark profile for the exact recipe fingerprint
and operating point.

## DP4 layout on an eight-device host

`llmd-values-dp4-on-x8.json` describes TP1/DP4/EP on an eight-device host. The
audit reports four engine ranks and four unassigned physical devices. A caller
must confirm whether the host is oversized, the DP setting is too low, or an
external launcher owns topology that the imported values do not describe.

## Cross-vendor DP8 shapes

The NVIDIA and AMD DP8 fixtures use the same sanitized model identity with
different runtime images, context limits, utilization settings, and
flow-control budgets. The draft importer preserves those differences, but the
structural audit does not rank the vendors. A comparison needs one pinned model
manifest plus initialization and serving measurements from each exact variant.

## What remains unresolved

The imported configuration drafts remain `incomplete`. The llm-d values
fixtures intentionally omit:

- an immutable model revision;
- exact architecture and quantization facts;
- measured resident weight bytes per device;
- measured runtime and activation reserves;
- measured context-bound KV capacity; and
- throughput and latency at a defined operating point.

Adding approximate download sizes or guessed throughput would make these cases
look complete while making the answer less trustworthy. ICC keeps those gaps
visible until a pinned model manifest, vLLM initialization profile, and exact
benchmark profile are attached.

The long-context fixture also has sanitized manifest and initialization inputs
that exercise the complete materialization path. Their numbers are internally
consistent test data, not measurements of a named public or private model.
