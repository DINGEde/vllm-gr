# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import json
import os
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

pytestmark = pytest.mark.cpu_test

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "online_serving"
CLIENT = EXAMPLES / "beam_search" / "client.sh"
VERIFIER = EXAMPLES / "catalog" / "verify_catalog.py"
SID_TOKENS = ["<|sid_begin|>", "<s_a_1>", "<s_b_2>", "<s_c_3>", "<|sid_end|>"]
SID = "".join(SID_TOKENS)


@pytest.mark.l1  # type: ignore[misc]
def test_grpc_client_accepts_a_short_catalog_result_and_checks_its_shape() -> None:
    pytest.importorskip("grpc", minversion="1.84.0")
    pytest.importorskip("google.protobuf", minversion="7.35.1")
    from examples.online_serving.grpc.client import generate
    from vllm_gr.entrypoints.grpc.generated import vllm_engine_pb2 as pb

    class Stub:
        offsets: tuple[int, ...] = (0, 3)
        scores: tuple[float, ...] = (-1.0,)
        flags: tuple[bool, ...] = (True,)

        def GenerateBatch(self, request: Any, *, timeout: float) -> Any:
            assert timeout > 0
            assert request.requests[0].beam_search.beam_width == 128
            return pb.BatchGenerateResponse(
                batch_id=request.batch_id,
                items=[
                    pb.BatchGenerateItem(
                        request_id=request.requests[0].request_id,
                        beams=pb.BeamBatchOutput(
                            output_ids=[4, 5, 6],
                            output_offsets=self.offsets,
                            cumulative_logprobs=self.scores,
                            has_cumulative_logprob=self.flags,
                        ),
                    )
                ],
            )

    stub = Stub()
    assert generate(stub, [[1]], width=128)["items"] == [
        {"token_ids": [[4, 5, 6]], "scores": [-1.0]}
    ]
    for offsets in ((1, 3), (0, 2), (0, 3, 2, 3)):
        stub.offsets = offsets
        with pytest.raises(ValueError, match="incomplete beam output"):
            generate(stub, [[1]], width=128)
    stub.offsets = (0, 3)
    stub.scores = ()
    with pytest.raises(ValueError, match="incomplete beam output"):
        generate(stub, [[1]], width=128)
    stub.scores = (-1.0,)
    stub.flags = ()
    with pytest.raises(ValueError, match="incomplete beam output"):
        generate(stub, [[1]], width=128)


@pytest.mark.parametrize("status", [200, 404])  # type: ignore[misc]
def test_beam_client_sends_valid_request_from_other_directory(tmp_path: Path, status: int) -> None:
    bash = shutil.which("bash")
    if bash is None or shutil.which("curl") is None:
        pytest.skip("The example requires Bash and curl")
    assert bash is not None

    received: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received["path"] = self.path
            received["auth"] = self.headers.get("Authorization")
            received["payload"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response: dict[str, object] = (
                {"choices": [{"text": SID}]} if status == 200 else {"error": "model not found"}
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

    prompt = 'A "quoted" path C:\\example\r\n\u63a8\u8350\n'
    prompt_path = tmp_path / "query with spaces.txt"
    prompt_path.write_bytes(prompt.encode("utf-8"))
    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            env = {
                **os.environ,
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                "ONEREC_BASE_URL": f"http://127.0.0.1:{server.server_port}/",
                "ONEREC_SERVED_MODEL_NAME": "onerec" if status == 200 else "wrong-model",
                "VLLM_API_KEY": "test-key",
            }
            cmd = [bash, CLIENT.as_posix(), "-p", prompt_path.as_posix()]
            completed = subprocess.run(
                cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15, check=False
            )
        finally:
            server.shutdown()
            thread.join()

    assert received["path"] == "/v1/completions", completed.stderr
    assert received["auth"] == "Bearer test-key"
    assert received["payload"] == {
        "model": env["ONEREC_SERVED_MODEL_NAME"],
        "prompt": prompt,
        "use_beam_search": True,
        "n": 128,
        "temperature": 0.0,
        "max_tokens": 5,
        "ignore_eos": True,
        "vllm_xargs": {"begin_token": "<|sid_begin|>", "end_token": "<|sid_end|>"},
    }
    assert (completed.returncode == 0) == (status == 200)
    assert ("choices" if status == 200 else "error") in json.loads(completed.stdout)


@pytest.mark.parametrize(  # type: ignore[misc]
    ("response", "trailer", "count", "exit_code"),
    [
        ({"choices": [{"text": SID}]}, "", 1, 0),
        ({"choices": [{"message": {"content": SID}}]}, " total time:0.35s", 1, 0),
        ({"error": {"message": "bad request"}}, "", 1, 1),
        ({}, "", 1, 1),
        ({"choices": []}, "", 1, 1),
        ({"choices": [{"message": {"content": ""}}]}, "", 1, 1),
        ({"choices": [{"text": "<s_a_1>"}]}, "", 1, 1),
        ({"choices": [{"text": SID.replace("<s_c_3>", "<s_c_4>")}]}, "", 1, 1),
        ({"choices": [{"text": SID}]}, "", 2, 1),
    ],
)
def test_catalog_verifier_reports_failure_for_invalid_results(
    tmp_path: Path, response: dict[str, object], trailer: str, count: int, exit_code: int
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(response) + trailer, encoding="utf-8")
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps([SID_TOKENS]), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            str(result_path),
            str(catalog_path),
            "--expected-count",
            str(count),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == exit_code, completed.stderr
    assert ("SUCCESS:" in completed.stdout) == (exit_code == 0)
    assert ("FAILURE:" in completed.stderr) == (exit_code != 0)
