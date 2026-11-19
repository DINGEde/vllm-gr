# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check OpenOneRec accuracy requests and failure reports."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    ("failure", "expected"),
    [
        ("http", "HTTPError"),
        ("timeout", "Timeout"),
        ("no choices", "no choices"),
        ("too few choices", "expected 2 choices"),
        ("bad choice", "no text message"),
        ("null content", "no text message"),
        ("empty content", "no text message"),
        ("whitespace only", "no text message"),
    ],
)
def test_accuracy_cli_reports_failed_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str, expected: str
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    def post(_url: str, **kwargs: Any) -> Any:
        assert kwargs["timeout"] == 12
        if failure == "timeout":
            raise requests.Timeout("read timed out")
        response = requests.Response()
        response.status_code = 500 if failure == "http" else 200
        choices = {
            "no choices": [],
            "too few choices": [{"message": {"content": "short"}}],
            "bad choice": [{}, {}],
            "null content": [
                {"message": {"content": "valid answer"}},
                {"message": {"content": None}},
            ],
            "empty content": [
                {"message": {"content": "", "reasoning_content": None}},
                {"message": {"content": "valid answer"}},
            ],
            "whitespace only": [
                {"message": {"content": " ", "reasoning_content": "\n"}},
                {"message": {"content": "valid answer"}},
            ],
        }
        response._content = json.dumps({"choices": choices.get(failure, [])}).encode()
        return response

    def load(**kwargs: Any) -> dict[str, dict[str, str]]:
        return {"video-42": {"prompt": "question", "ground_truth": "answer"}}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(accuracy, "DataLoaderWrapper", lambda **kw: SimpleNamespace(load_data=load))
    monkeypatch.setattr(accuracy.requests, "post", post)
    monkeypatch.setitem(
        accuracy.TASK_REGISTRY["video"].config["evaluation_config"], "evaluation_mode", "sid"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--endpoint",
            "/v1/chat/completions",
            "--n",
            "2",
            "--request-timeout",
            "12",
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 1
    report = json.loads((tmp_path / "video-failed.json").read_text())
    assert report["status"] == "failed"
    assert expected in report["sample_errors"]["video-42"]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_accuracy_keeps_successful_wrong_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from transformers import AutoTokenizer

    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    messages = [
        {"role": "system", "content": "Preserve whitespace."},
        {"role": "user", "content": "\nquestion\n"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Next question"},
    ]
    data = {
        "video-42": {
            "prompt": "locally rendered prompt",
            "chat_messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "ground_truth": "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>",
        }
    }

    def post(_url: str, **kwargs: Any) -> Any:
        assert kwargs["json"]["messages"] == messages
        assert kwargs["json"]["chat_template_kwargs"] == {"enable_thinking": True}
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {"message": {"content": "<|sid_begin|><s_a_4><s_b_5><s_c_6><|sid_end|>"}}
                ]
            },
        )

    reg = accuracy.TASK_REGISTRY["video"]
    config = {
        **reg.config,
        "evaluation_config": {"metrics": ["pass@1"], "k_values": [1], "evaluation_mode": "sid"},
    }
    monkeypatch.setitem(
        accuracy.TASK_REGISTRY,
        "video",
        SimpleNamespace(config=config, evaluator_class=reg.evaluator_class),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(
        accuracy,
        "DataLoaderWrapper",
        lambda **kw: SimpleNamespace(load_data=lambda **kw: data),
    )
    monkeypatch.setattr(accuracy.requests, "post", post)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "video",
            "--endpoint",
            "/v1/chat/completions",
            "--n",
            "1",
            "--chat-template-kwargs",
            '{"enable_thinking": true}',
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 0
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert report["total_samples"] == 1
    assert report["pass@1"] == 0


@pytest.mark.parametrize(  # type: ignore[misc]
    "message,expected",
    [
        ({"reasoning": "a clue", "content": None}, "a clue"),
        ({"reasoning_content": "old clue", "content": None}, "old clue"),
        ({"reasoning": "A", "reasoning_content": "A", "content": "B"}, "AB"),
        ({"reasoning": "new", "reasoning_content": "old", "content": " answer"}, "new answer"),
        ({"reasoning": None, "reasoning_content": "old", "content": " answer"}, "old answer"),
        ({"reasoning": "", "reasoning_content": "old", "content": None}, "old"),
        ({"content": "answer"}, "answer"),
        ({"content": None, "reasoning": None, "reasoning_content": ""}, None),
        ({"reasoning": 3, "content": "answer"}, None),
        ({"reasoning_content": [], "content": "answer"}, None),
        ({"content": {"text": "answer"}}, None),
    ],
)
def test_accuracy_http_selects_reasoning_field(
    monkeypatch: pytest.MonkeyPatch, message: dict[str, Any], expected: str | None
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    response = SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {"choices": [{"message": message}]},
    )
    monkeypatch.setattr(accuracy.requests, "post", lambda *a, **kw: response)
    if expected is None:
        with pytest.raises(accuracy.InferenceError, match="no text message|non-text"):
            accuracy.run_api_inference(["prompt"], "model", "http://localhost/unused", "EMPTY", {})
    else:
        outputs = accuracy.run_api_inference(
            ["prompt"], "model", "http://localhost/unused", "EMPTY", {}
        )
        assert outputs[0].outputs[0].text == expected


def test_accuracy_classification_requires_logprobs_before_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import benchmarks.open_one_rec.one_rec_acc_test as accuracy

    def unexpected_load(**kwargs: Any) -> Any:
        pytest.fail("classification attempted to load input without logprob support")

    monkeypatch.setattr(accuracy, "DataLoaderWrapper", unexpected_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--task-type",
            "label_pred",
            "--endpoint",
            "/v1/chat/completions",
            "--result-dir",
            str(tmp_path),
        ],
    )
    assert accuracy.main() == 1
    report = json.loads((tmp_path / "label_pred-failed.json").read_text())
    assert report["auc"] is None
    assert report["valid_samples"] == report["invalid_samples"] == 0
    assert report["auc_unavailable_reason"] == "classification_logprobs_unavailable"
