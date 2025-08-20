# Quickstart

## Prerequisites

### Supported Device

* Enflame **S60**
* Compatible servers/workstations with GCU integration

> 🔧 Make sure you've installed the **TopsRider_i3x 3.4+** runtime stack and have validated GCU using `efsmi`.

---

## Usage

For faster downloads inside China, enable ModelScope:

```bash
export VLLM_USE_MODELSCOPE=true
```

You have two options to use vLLM-GCU:

---

### Option 1: Inference with Python interface

```python
from vllm import LLM, SamplingParams

prompts = [
    "Please talk about China in 100 words.",
    "Which country is now best for travelling?",
]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", device="gcu")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"> {output.prompt.strip()} → {output.outputs[0].text.strip()}")
```

> ⏱ It may takes few minutes to download the qwen2.5 0.5B model.

---

### Option 2: Running OpenAI-API Compatible Server

Start vLLM server for Qwen3 32B MoE model on two GCUs:

```bash
python3 -m vllm.entrypoints.openai.api_server \
 --model [folder of Qwen/Qwen3-Coder-30B-A3B-Instruct] \
 --tensor-parallel-size 2 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.9  \
 --trust-remote-code
```

Query the running models:

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

Send request(s) with curl or any OpenAI-API compatible clients

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "prompt": "Please talk about China in 100 words.",
    "max_tokens": 256,
    "temperature": 0
}' | python3 -m json.tool
```

---

### Stop the Server Gracefully

```bash
VLLM_PID=$(pgrep -f "vllm")
kill -2 "$VLLM_PID"
```

Or `Ctrl+C` to stop vllm service.
