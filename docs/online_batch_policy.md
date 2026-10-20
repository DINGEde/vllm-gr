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

On NPU, batches requiring Decode must also satisfy
`batch_size * beam_width <= max_num_batched_tokens`, because Decode expands
each logical input into physical rows in native input buffers. The planner
splits further when needed; this is separate from the Prefill token sum and
does not apply to CUDA or to a generation budget of one (Prefill-only).
An input that cannot fit even alone rejects the whole plan before submission.

With compatible inputs and sufficient capacity:

| Policy | Input count | Execution sizes |
| --- | --- | --- |
| Default `[1, 2, 4]` | 7 | 4 + 2 + 1 |
| `[1, 3]` | 7 | 3 + 3 + 1 |
| `[1, 3, 7]` | 10 | 7 + 3 |
| `[1, 3, 7]` | 8 | 7 + 1 |
| `[1, 2, 3, 4, 5, 6, 7, 8]` | 8 | 8 |

Only listed sizes are used. With `[1, 3, 7]` and `max_num_seqs=4`, seven inputs
become `3 + 3 + 1`, not `4 + 3`. The existing batch-plan INFO log reports the
actual sizes. Output order, serial cleanup and cancellation semantics are
unchanged. Direct offline/engine batch calls are not repartitioned by this policy.

## Performance Boundary

This option controls online grouping, not CUDA/NPU graph capture or persistent
workspace allocation. Enabling a size does not guarantee graph replay for it;
uncaptured Prefill shapes may execute eagerly. Verify actual execution mode,
warmup cost, memory use and end-to-end latency before selecting a deployment
policy. Largest-first packing is not a throughput-optimal search. Changing the
policy can change numerical outputs because the execution shape changes.
