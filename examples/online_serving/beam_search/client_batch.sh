#!/usr/bin/env bash
# Submit k prompts in one completions request. V1 partitions them into
# contiguous batches of 4, 2, or 1 and executes one session at a time.
#
# Requires a V1 server (see README.md for the launch command) started with
# enough sequence and token capacity for each execution batch. Smaller limits
# produce smaller batches; each prompt must fit individually. The effective
# generation budget must agree within each batch.
#
# k may be any positive integer. The width defaults to 128 because V1 fixes the beam
# width at startup and supports at most 256.

bw=128
k=4
while getopts "b:k:" opt; do
  case $opt in
    b) bw=$OPTARG ;;
    k) k=$OPTARG ;;
    *) echo "only -b <beam_width> and -k <prompt_count> are supported"; exit 1 ;;
  esac
done

if ! [[ "$k" =~ ^[1-9][0-9]*$ ]]; then
  echo "-k must be a positive integer"; exit 1
fi

base=\"$(cat ../../../tests/resources/single_one_rec_prompt.txt)\"

# Each slot needs its own prompt. Identical prompts share one Prefix Cache
# entry, so the batch would measure a cache hit instead of the cache-miss path
# this example exists to show. A per-slot marker just after the user turn opens
# keeps the k prompts distinct without changing the request shape.
prompts=""
for i in $(seq 0 $((k - 1))); do
  slot=${base/<|im_start|>user/<|im_start|>user [slot $i]}
  if [ -z "$prompts" ]; then
    prompts="$slot"
  else
    prompts="$prompts, $slot"
  fi
done

curl -w " total time:%{time_total}s\n" -X POST "http://localhost:8000/v1/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "OpenOneRec/OneRec-1.7B",
		"prompt": ['"$prompts"'],
        "use_beam_search": true,
         "n": '"$bw"',
        "temperature": 0.0,
        "max_tokens": 5,
        "ignore_eos": false,
		    "vllm_xargs": {
            "begin_token": "<|sid_begin|>",
            "end_token": "<|sid_end|>"
        }
	}'
