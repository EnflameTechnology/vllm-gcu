# Using EvalScope

This document will guide you have model inference stress testing and accuracy testing using [EvalScope](https://github.com/modelscope/evalscope).

## 1. Online serving

You can run docker container to start the vLLM server on a single gcu:

```{code-block} bash
   :substitutions:
python3 -m vllm.entrypoints.openai.api_server \
 --model Qwen/Qwen2.5-7B-Instruct \
 --tensor-parallel-size 1 \
 --max-model-len 26240 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code
```

Once the server is started, you can query the model with given prompts:

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The popular LLM models are",
        "max_tokens": 256,
        "temperature": 0
    }'
```

## 2. Install EvalScope

```bash
pip install gradio plotly evalscope
```

## 3. Run gsm8k accuracy test with EvalScope

You can `evalscope eval` run gsm8k accuracy test:
```
evalscope eval \
 --model Qwen/Qwen2.5-7B-Instruct \
 --api-url http://localhost:8000/v1 \
 --api-key EMPTY \
 --eval-type service \
 --datasets gsm8k \
 --limit 20
```

The sample evaluation results:

```shell
+---------------------+-----------+-----------------+----------+-------+---------+---------+
| Model               | Dataset   | Metric          | Subset   |   Num |   Score | Cat.0   |
+=====================+===========+=================+==========+=======+=========+=========+
| Qwen2.5-7B-Instruct | gsm8k     | AverageAccuracy | main     |    20 |     0.8 | default |
+---------------------+-----------+-----------------+----------+-------+---------+---------+
```

More details can be found: [EvalScope](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#model-api-service-evaluation).

## 4. Run stress test with EvalScope

Install evalscope performance test package:

```shell
pip install evalscope[perf] -U
```

Run `evalscope perf` for performance evaluation:

```
evalscope perf \
    --url "http://localhost:8000/v1/chat/completions" \
    --parallel 10 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --number 50 \
    --api openai \
    --dataset openqa \
    --stream
```

### Sample results

```shell
Benchmarking summary:
+-----------------------------------+---------------------------------------------------------------+
| Key                               | Value                                                         |
+===================================+===============================================================+
| Time taken for tests (s)          | 25.2652                                                       |
+-----------------------------------+---------------------------------------------------------------+
| Number of concurrency             | 10                                                             |
+-----------------------------------+---------------------------------------------------------------+
| Total requests                    | 40                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Succeed requests                  | 40                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Failed requests                   | 0                                                             |
+-----------------------------------+---------------------------------------------------------------+
| Output token throughput (tok/s)   | 279.5536                                                      |
+-----------------------------------+---------------------------------------------------------------+
| Total token throughput (tok/s)    | 355.4561                                                      |
+-----------------------------------+---------------------------------------------------------------+
| Request throughput (req/s)        | 0.5212                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average latency (s)               | 8.3612                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average time to first token (s)   | 0.1533                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average time per output token (s) | 0.0196                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average igcut tokens per request  | 50.25                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Average output tokens per request | 364.4                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Average package latency (s)       | 0.0154                                                        |
+-----------------------------------+---------------------------------------------------------------+
| Average package per request       | 254.6                                                         |
+-----------------------------------+---------------------------------------------------------------+
| Expected number of requests       | 40                                                            |
+-----------------------------------+---------------------------------------------------------------+
| Result DB path                    | outputs/Qwen2.5-7B-Instruct/benchmark_data.db |
+-----------------------------------+---------------------------------------------------------------+
```

More details for stress test: [EvalScope](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html#basic-usage).
