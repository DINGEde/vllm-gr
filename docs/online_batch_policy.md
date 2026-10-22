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
  online_batch_sizes: [1, 3, 7]
```

This is a fragment, not a complete deployment configuration. Keep the existing
model, constraint-table and other beam settings. Set `--max-num-seqs` high
enough for the desired size, for example `8` to admit B=7 or B=8.
The legacy additional-config equivalent is
`"vllm_gr_beam_online_batch_sizes": [1, 3, 7]`; do not mix canonical and legacy
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
| `[1, 3, 7]` | 10 | 7 + 3 |
| `[1, 3, 7]` | 8 | 7 + 1 |
| `[1, 2, 3, 4, 5, 6, 7, 8]` | 8 | 8 |

Only listed sizes are used. A non-default policy exceeding `max_num_seqs`
is rejected at engine configuration. With `[1, 3]` and `max_num_seqs=4`, seven
inputs become `3 + 3 + 1`, not `4 + 3`. Output order, serial cleanup and
cancellation semantics are unchanged. Direct offline/engine batch calls are not
repartitioned by this policy.

Every plan logs the executed `sizes` together with `configured_sizes`,
`usable_sizes` and `reasons`. `usable_sizes` is what capacity leaves after
`max_num_seqs` and the NPU Decode row cap; `reasons` names the static limits
that kept the plan narrow: `unequal_effective_budgets`, `sequence_capacity`,
`configured_singletons` and `token_budget`. A list that ends up all singletons
logs the same fields. When configuration and capacity would have allowed a
larger batch, an all-singleton plan also emits one WARNING at most every five
minutes, carrying the number of suppressed warnings since the last one. A
configured policy of `[1]` or a capacity of one never warns.

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
`ServiceUnavailableError` naming the rejected sub-batch. Earlier sub-batches are
not returned as partial output, and no sub-batch is retried: choosing the
grouping was the server's decision, so a refused group is not reported as an
invalid caller request.

## Performance Boundary

This policy is provided by the fixed-batch foundation PR. It also selects
default CUDA V1/V2 Prefill request counts and V2 Decode startup capture sizes.
The Prefill-only environment override remains independent; V1 Decode and NPU
exact Decode graph settings retain their existing semantics. Persistent
workspace capacity is independent of this policy. Enabling a size does not
guarantee graph replay for every shape. See [capture configuration](prefill_graph_capture.md).
Verify actual execution mode,
warmup cost, memory use and end-to-end latency before selecting a deployment
policy. Largest-first packing is not a throughput-optimal search. Changing the
policy can change numerical outputs because the execution shape changes.
