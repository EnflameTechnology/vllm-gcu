# Single GCU (Qwen3 8B)

### Offline Inference on Single GCU

Run the following script to execute offline inference on a single gcu:

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
        model="Qwen/Qwen3-8B",
        max_model_len=26240
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
        model="Qwen/Qwen3-8B",
        max_model_len=26240,
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

### Online Serving on Single GCU

Run docker container to start the vLLM server on a single GCU:

:::::{tab-set}
::::{tab-item} Graph Mode

```{code-block} bash
   :substitutions:
python3 -m vllm.entrypoints.openai.api_server \
 --model [Qwen3-8B folder] \
 --tensor-parallel-size 1 \
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
 --model [Qwen3-8B folder] \
 --tensor-parallel-size 1 \
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
The parameter `--max_model_len` should not be larger than the maximum number of tokens that can be stored in KV cache.
:::

Once your server is started, you can query the model with the given prompt(s):

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-8B",
        "prompt": "The largest continent in the world is",
        "max_tokens": 256,
        "temperature": 0
    }'
```