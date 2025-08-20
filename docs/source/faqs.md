# FAQs

## General FAQs

### 1. What devices are currently supported?

vLLM-GCU currently supports:

* Enflame **S60** Inference Card
* Enflame-compatible GCU systems running with **TopsRider ≥ i3x 3.4**

> Check availability using `efsmi`.

---

### 2. Where can I get vLLM-GCU Docker images?

You can pull from the \[Enflame Docker Registry] or directly from [DockerHub](https://hub.docker.com/r/enflame/vllm-gcu) (if made public).

Inside China, you can accelerate with `daocloud` or private mirrors:

```bash
docker pull registry.daocloud.io/enflame/vllm-gcu:latest
```

---

### 3. What models are supported on GCU?

vLLM-GCU supports most Hugging Face-compatible transformer models, including:

* `Qwen2.5` and `Qwen3` series
* `DeepSeek V3`
* `Baichuan`
* `LLaMA2` & `LLaMA3`
* `ChatGLM`
* `InternLM`

For full compatibility and graph-mode acceleration, see [Model Support Matrix (GCU)](user_guide/support_matrix/supported_models.md).

---

### 4. What features does vLLM-GCU support?

* OpenAI-compatible REST API
* Graph-mode execution on GCU
* FP16 / BF16 precision
* 4-bit and 8-bit quantization
* KV Cache for efficient decoding
* Tensor/Pipeline parallelism on multi-card setups
* Chunked Prefill
* Static/dynamic batch (continuous batching)
* Offline batch inference

---

### 5. How is the performance on GCU?

vLLM-GCU accelerates models like `Qwen3`, `LLaMa3`, `GLM4` and `DeepSeek` by using graph-mode execution by default. You can disable graph-mode by setting:

```shell
--enforce-eager=True
```

Additionally, large models can be inferenced with multi-gcus by setting `--tensor-parallel-size`.

---

### 6. How does vLLM-GCU integrate with vLLM?

vLLM-GCU is implemented as a **plugin** to vLLM using Enflame’s GCU kernel interface. Make sure vLLM-GCU and vLLM use **matching versions** (e.g. `vllm==0.8.0`, `vllm-gcu==0.8.0`).

---

### 7. Does vLLM-GCU support quantization?

Yes. vLLM-GCU supports both **GPTQ** (8-bit) and **AWQ** (4-bit/8-bit) quantizations using GCU-optimized kernels.

---

### 8. How to run DeepSeek and Qwen3 MoE models?

Follow the model tutorial by `vllm serve` Qwen3 MoE models directly or refer to [Multi-Node Inference](tutorials/multi_node.md) for DeepSeek models.

---

### 9. How to avoid Out Of Memory (OOM) on GCU?

* Reduce `--gpu-memory-utilization` (default is `0.9`) or enable chunked prefill. 

---

### 10. Can't reinstall vllm-gcu after uninstall?

Use:

```bash
python setup.py clean
python setup.py install
```

Or:

```bash
pip install --no-cache-dir .
```

---

### 11. How to ensure deterministic results?

Use greedy decoding and set sampling parameter `temperature` to zero:

```python
SamplingParams(temperature=0)
```

---

### 12. How to contact the vLLM-GCU community?

* [GitHub Issues](https://github.com/enflame-tech/vllm-gcu/issues)
* Join our **WeChat dev group** (QR code in the repo)
* Follow **weekly office hours** via Tencent Meeting (announced on GitHub)
* Discuss in [vLLM Forum GCU Channel](https://discuss.vllm.ai)
