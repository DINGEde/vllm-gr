# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Send contiguous int64 token buffers through the GR GenerateBatch RPC."""

import argparse
import json
import uuid
from pathlib import Path

import grpc
import numpy as np

from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb
from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2_grpc as rpc


def generate(stub, rows, *, width=4, max_new_tokens=3):
    batch_id = uuid.uuid4().hex
    requests = []
    for index, row in enumerate(rows):
        tokens = np.asarray(row, dtype="<i8")
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError("each prompt must be a nonempty one-dimensional token array")
        requests.append(
            pb.GenerateRequest(
                request_id=f"{batch_id}-{index}",
                tokenized=pb.TokenizedInput(input_ids_i64_le=tokens.tobytes()),
                sampling_params=pb.SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    ignore_eos=True,
                    n=width,
                ),
                beam_search=pb.BeamSearchOptions(beam_width=width),
            )
        )
    response = stub.GenerateBatch(
        pb.BatchGenerateRequest(batch_id=batch_id, requests=requests),
        timeout=120,
    )
    if response.batch_id != batch_id or len(response.items) != len(requests):
        raise ValueError("response batch identity or cardinality mismatch")
    items = []
    for index, item in enumerate(response.items):
        if item.request_id != requests[index].request_id:
            raise ValueError("response item order changed")
        beams = item.beams
        flat, offsets = list(beams.output_ids), list(beams.output_offsets)
        count = len(offsets) - 1
        scores = list(beams.cumulative_logprobs)
        if (
            not 1 <= count <= width
            or offsets[0] != 0
            or offsets[-1] != len(flat)
            or any(a >= b for a, b in zip(offsets, offsets[1:]))
            or len(scores) != count
            or len(beams.has_cumulative_logprob) != count
            or not all(beams.has_cumulative_logprob)
        ):
            raise ValueError("incomplete beam output")
        items.append(
            dict(
                token_ids=[flat[a:b] for a, b in zip(offsets, offsets[1:])],
                scores=scores,
            )
        )
    return dict(batch_id=batch_id, items=items)


def main():
    parser = argparse.ArgumentParser(
        description="Send a OneRec prompt or random token IDs for a tiny-model smoke test."
    )
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--model", help="Local OneRec snapshot used by the server")
    parser.add_argument("--prompt", type=Path, help="UTF-8 OneRec prompt file")
    args = parser.parse_args()
    if bool(args.model) != bool(args.prompt):
        parser.error("--model and --prompt must be provided together")
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        begin_id, end_id = tokenizer.convert_tokens_to_ids(["<|sid_begin|>", "<|sid_end|>"])
        prompt = args.prompt.read_text(encoding="utf-8")
        rows = [
            [*tokenizer.encode(prompt, add_special_tokens=False), begin_id]
            for _ in range(args.batch_size)
        ]
    else:
        rows = np.random.default_rng(7).integers(
            3,
            args.vocab_size,
            size=(args.batch_size, args.tokens),
            dtype=np.int64,
        )
    with grpc.insecure_channel(args.address) as channel:
        result = generate(
            rpc.VllmEngineStub(channel), rows, width=args.width, max_new_tokens=args.max_new_tokens
        )
    if args.model:
        result["choices"] = [
            {"text": tokenizer.decode([begin_id, *ids, end_id], skip_special_tokens=False)}
            for item in result["items"]
            for ids in item["token_ids"]
        ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
