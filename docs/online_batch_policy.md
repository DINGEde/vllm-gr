# Online Beam Batch Policy

The V1 engine admits fixed-membership logical batches of 1 through 8 inputs.
The non-streaming beam completions adapter independently chooses which sizes
to use when partitioning a prompt list. Its default remains `[1, 2, 4]`.
One session executes at a time; this option does not enable concurrent owners
or combine inputs from different HTTP requests.

## Configuration

Add `online_batch_sizes` to the `beam` section of an existing canonical GR
configuration passed through `--vllm-gr-config`:

```yaml
beam:
  execution_mode: v1
  online_batch_sizes: [1, 4, 7]
```

This is a fragment, not a complete deployment configuration. Keep the existing
model, constraint-table and other beam settings. Set `--max-num-seqs` high
enough for the desired size, for example `8` to admit B=7 or B=8.
The legacy additional-config equivalent is
`"vllm_gr_beam_online_batch_sizes": [1, 4, 7]`; do not mix canonical and legacy
GR configuration keys. No per-request protocol field is added.

The value must be a nonempty list of distinct integers in `[1, 8]` containing
`1`. Missing singleton support, duplicates, booleans, fractional values and
unsupported sizes fail configuration validation. Values are sorted internally;
list order does not express priority. Omitting the option preserves `[1, 2, 4]`.

## Planning Semantics

The planner chooses the largest legal contiguous batch at each position.
Members must have equal effective generation budgets, fit the scheduling token
budget together, and satisfy `max_num_seqs`. A configured size is eligible,
not guaranteed: capacity and per-input budgets can require smaller batches.

With compatible inputs and sufficient capacity:

| Policy | Input count | Execution sizes |
| --- | --- | --- |
| Default `[1, 2, 4]` | 7 | 4 + 2 + 1 |
| `[1, 3]` | 7 | 3 + 3 + 1 |
| `[1, 4, 7]` | 11 | 7 + 4 |
| `[1, 4, 7]` | 8 | 7 + 1 |
| `[1, 2, 3, 4, 5, 6, 7, 8]` | 8 | 8 |

Only listed sizes are used. A non-default policy exceeding `max_num_seqs`
is rejected at engine configuration. With `[1, 3]` and `max_num_seqs=4`, seven
inputs become `3 + 3 + 1`, not `4 + 3`. Output order, serial cleanup and
cancellation semantics are unchanged. Direct offline/engine batch calls are not
repartitioned by this policy.

Every plan logs one line: `batches` lists each sub-batch as
`(start, stop, size, boundary_reason)`, so the reason a given sub-batch is a
singleton is readable without inferring it from the list as a whole.
`usable_sizes` is what capacity leaves after `max_num_seqs` and the NPU Decode
row cap, and `capacity_source` names which of the two did the cutting
(`max_num_seqs` or `decode_rows`) when the usable width is below the configured
width; it is `None` when the deployment runs the configured width.
`boundary_reason` and `reasons` use one vocabulary:
`unequal_effective_budgets`, `token_budget` and `configured_singletons`. They
name limits that held the plan below `usable_sizes`; a deployment capacity below
the configured width is reported by `capacity_source`, not as a reason, because
a plan whose batches all reach `usable_sizes` has lost no width. A list that
ends up all singletons logs the same fields. When configuration and capacity
would have allowed a larger batch, an all-singleton plan also emits one WARNING
at most every five minutes, carrying the number of suppressed warnings since the
last one; the interval and the counter are per process, so a deployment with N
workers warns up to N times per interval. A configured policy of `[1]` or a
capacity of one never warns.

## Failure Reporting

Every input is validated before the first sub-batch is submitted: prompt
length, the scheduling token budget, `max_num_seqs`, and the effective
generation budget after begin/end tokens are removed from `max_tokens`. The
latter must leave 1 to 3 Decode steps per input and must not exceed the
configured `beam_max_decode_steps`. A rejection the planner can
decide statically is therefore answered as a caller error before anything runs,
so no prompt is computed without a place for its answer.

Device and KV admission remain the engine's decision. If the engine refuses a
sub-batch this adapter assembled, the request is answered with HTTP 503
`ServiceUnavailableError` naming the rejected sub-batch and how many sub-batches
had already completed. Earlier sub-batches are not returned as partial output,
and no sub-batch is retried: choosing the grouping was the server's decision, so
a refused group is not reported as an invalid caller request. The refusal is
therefore asymmetric in cost: a k-input list refused at sub-batch j has already
computed j sub-batches whose output is dropped, and retrying recomputes all of
them. A caller that retries automatically should back off as it would for any
503.

## Rollback

Set `VLLM_GR_DISABLE_ONLINE_BATCH_SPLIT` to `1`, `true`, `yes` or `on` to take
the split path out of service without a code change or a restart: the variable
is read on every request, and requests then take the upstream handler exactly as
before the feature. Any other value, including an empty one, leaves the path
armed. Reverting the code remains the only way to remove the patch itself.

Configuring `online_batch_sizes: [1]` is not a rollback. It keeps every request
on this path -- rendered once here, partitioned here, identifiers assigned here
-- and only makes each sub-batch hold one input.

## Performance Boundary

This policy is provided by the fixed-batch foundation PR. It also selects
default CUDA V1/V2 Prefill request counts and V2 Decode startup capture sizes.
The Prefill-only environment override remains independent; V1 Decode and NPU
exact Decode graph settings retain their existing semantics. Persistent
workspace capacity is independent of this policy. Enabling a size does not
guarantee graph replay for every shape. See [capture configuration](prefill_graph_capture.md).
Verify actual execution mode,
warmup cost, memory use and end-to-end latency before selecting a deployment
policy. Largest-first packing is not a throughput-optimal search, and it is not
uniformly faster than sending the same prompts as separate requests. Measured on
one L20 with `--max-num-seqs 8`, Prefix Cache off, beam width 128 and 1024-token
prompts, each k-prompt list against the same k prompts sent one at a time, under
the default `[1, 2, 4]` policy:

| k | Plan | List (ms) | Separate (ms) | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 2 | 2 | 109.6 | 131.7 | 1.20 |
| 4 | 4 | 210.5 | 265.6 | 1.26 |
| 8 | 4 + 4 | 421.0 | 531.7 | 1.26 |
| 3 | 2 + 1 | 463.3 | 197.2 | 0.42 |
| 6 | 4 + 2 | 580.9 | 397.7 | 0.67 |
| 7 | 4 + 2 + 1 | 786.8 | 464.9 | 0.59 |

Plans that use one width were 1.2 to 1.3 times faster; plans that mix widths
were up to 2.4 times slower. The extra cost is in the mixing, not in the number
of sub-batches: under `[1, 4, 7]`, a `7 + 7` plan cost 704.6 ms against 357.1 ms
for a single 7 -- twice the width, twice the time, no penalty -- while `7 + 1`
and `7 + 4` were slower than the same prompts sent separately (683.4 ms against
528.7 ms, and 803.2 ms against 727.1 ms). The Prefill capture grid is not the
cause: naming `VLLM_GR_PREFILL_GRAPH_CAPTURE_NUM_REQS=1,2,4` explicitly left the
mixed plans at the same speed. Beyond that the cost is not localized here, so
measure the list lengths your traffic actually sends rather than assuming a
wider policy is a faster one.

Changing the policy can also change numerical outputs, because the execution
shape changes. At temperature 0, on 128 video-task prompts, a 4-input batch and
the same prompts sent one at a time agreed on the top-ranked candidate for 110
of them (85.9%), with a mean top-32 overlap of 29.8 of 32. Two runs of the
single-prompt arm agreed on all 128, so the disagreement comes from the batch
shape rather than from run-to-run variance. A policy that batches by default therefore returns
answers that a per-input `[1]` deployment does not.

A k-input request runs its sub-batches serially, but it does not hold the engine
between them: requests from other clients are admitted in the gaps. A k-input
request's end-to-end latency therefore includes whatever else the engine ran,
and its tail latency is not the sum of its own sub-batch latencies alone.
