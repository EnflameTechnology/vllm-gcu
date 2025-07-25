# Multi-GCU (Qwen2.5 32B)

### Offline Inference on Multiple GCUs

Run the following script to execute offline inference on multiple GCUs:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} python
   :substitutions:
import os
from vllm import LLM, SamplingParams

prompts = [
    "The best country for travelling is",
    "The largest continent in the world is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
        model="Qwen/Qwen2.5-32B",
        max_model_len=26240,
        tensor_parallel_size=4,
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
::::

::::{tab-item} Eager Mode

```{code-block} python
   :substitutions:
import os
from vllm import LLM, SamplingParams

prompts = [
    "The best country for travelling is",
    "The largest continent in the world is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
        model="Qwen/Qwen2.5-32B",
        max_model_len=26240,
        tensor_parallel_size=4,
        enforce_eager=True
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
::::
:::::

### Online Serving on Multiple GCUs

Run docker container to start the vLLM server on multiple GCUs:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} bash
   :substitutions:
python3 -m vllm.entrypoints.openai.api_server \
 --model [Qwen2.5-32B folder] \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code
```
::::

::::{tab-item} Eager Mode

```{code-block} bash
   :substitutions:
python3 -m vllm.entrypoints.openai.api_server \
 --model [Qwen2.5-32B folder] \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code \
 --enforce-eager
```
::::
:::::

:::{note}
`--tensor-parallel-size` controls the number of GCU uses, while, the parameter `--max_model_len` should not be larger than the maximum number of tokens that can be stored in KV cache.
:::

Once your server is started, you can query the model with the given prompt(s):

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-32B",
        "prompt": "The largest continent in the world is",
        "max_tokens": 256,
        "temperature": 0
    }'
```