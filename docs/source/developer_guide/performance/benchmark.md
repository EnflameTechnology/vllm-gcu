# Performance Benchmark

## 1. Download model weights (optional)
For faster running speed, we recommend downloading the model in advance：
```bash
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct
```

## 2. Run the benchmark

```bash
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
 --num-prompts 100 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
 ```
