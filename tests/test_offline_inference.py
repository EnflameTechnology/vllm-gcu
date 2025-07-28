"""Compare the short outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/test_offline_inference.py`.
"""
import os

import pytest
import vllm  # noqa: F401
from conftest import VllmRunner
from typing import Dict

MODEL_INF = [
    {
        'name': "/home/.cache/tops/dataset/inference/scorpio/vllm/qwen2.5-0.5b-instruct",
        'golden': " 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "
    }
]

@pytest.mark.parametrize("model_inf", MODEL_INF)
@pytest.mark.parametrize("dtype", ["float16"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_model(
    model_inf: Dict,
    dtype: str,
    max_tokens: int,
    tensor_parallel_size: int,
) -> None:

    # 5042 tokens for gemma2
    # gemma2 has alternating sliding window size of 4096
    # we need a prompt with more than 4096 tokens to test the sliding window
    prompt = "The following numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with VllmRunner(model_inf['name'],
                    max_model_len=8192,
                    dtype=dtype,
                    enforce_eager=False,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=0.7) as vllm_model:
        output = vllm_model.generate_greedy(example_prompts, max_tokens)
        generate = output[0][1][len(prompt):]
        assert generate == model_inf['golden']
