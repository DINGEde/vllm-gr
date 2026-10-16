# Beam search example

## Installation

refer to [quickstart.md](../../../docs/quickstart.md) for installation

## running example

- This example demonstrates using vllm-gr for executing generative recommendation system based OneRec-1.7B model

- Run server (beam search patches are auto-loaded via plugin)

```bash
vllm-gr serve OpenOneRec/OneRec-1.7B \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --max-logprobs 1024 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 16384 \
    --scheduling-policy priority
```

run client (on another terminal)

- the client execute a predefined 120 items prompt and ask for item recommendations.
- the server would reply with **beam_width** suggested items.

```bash
bash ./client.sh [-b beam_width ]

## Expected result

- sorted recommendations in json format with fields:
    - content: recommended token
- total execution time

```bash
./client.sh -b 512  
```

Results of vllm-gr looks like this

```json
{"id":"chatcmpl-a779679c38e57685","object":"chat.completion","created":1770297709,"model":"OpenOneRec/OneRec-1.7B","choices":
{"index":0,"message":{"role":"assistant","content":"<|sid_begin|><s_a_4113><s_b_7330><s_c_2009><|sid_end|>","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null,"reasoning_content":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null}...{"index":511...}}
total time: 1.610658s
```

- Example was demonstrated on:

    - Host: Intel(R) Xeon(R) Gold 5320T CPU @ 2.30GHz
    - Device: Nvidia RTX A6000 48GB

## Batching several prompts in one HTTP request

With `execution_mode: v1`, a prompt **list** sent to `POST /v1/completions` is
split into contiguous batches of 4, 2, or 1 and executed serially. With enough
capacity and equal generation budgets, 5 prompts run as `4+1` and 6 as `4+2`.
This uses the completions route, independently of the chat example above.

That route needs a V1 server rather than the Legacy one above. V1 fixes the beam
width at startup and supports at most 256, so the Legacy example's `-b 512` is
refused here, and it needs a GR config carrying a Constraint Table.

```bash
vllm-gr serve OpenOneRec/OneRec-1.7B \
    --max-logprobs 128 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 16384 \
    --no-enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --vllm-gr-config /path/to/vllm_gr_config.yaml
```

`beam.max_width` in that config has to be 128, matching this client's default.
[The V1 quickstart](../../../docs/onerec_v1_async_quickstart.md) describes the
config fields and the Constraint Table.

Then, on another terminal:

```bash
bash ./client_batch.sh [-b beam_width] [-k prompt_count]   # k is any positive integer
```

The script marks each prompt with a distinct slot to reduce shared-prefix
effects in measurements. Distinct prompts are not an API requirement, and
shared prefixes may still be cached.

The response is a single object whose `choices` are flat-indexed prompt-major,
so `k` prompts × `n` beams give `k × n` choices, and `usage.prompt_tokens` sums
across the batch.

Two server settings constrain each execution batch, not the full HTTP list:

| Setting | Requirement |
| --- | --- |
| `--max-num-seqs` | At least the execution batch size, up to 4 |
| `--max-num-batched-tokens` | At least its summed prompt lengths, including begin tokens |

Smaller capacity produces smaller execution batches. Different effective
generation budgets also split batches without shortening any prompt's output.
An individually invalid prompt remains an error. Sub-batch failure stops later
submissions; the response is returned only when the whole list succeeds. See
[the V1 quickstart](../../../docs/onerec_v1_async_quickstart.md#batching-several-prompts-in-one-http-request)
for capacity, cancellation, and response semantics.
