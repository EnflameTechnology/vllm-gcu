# FAQs

## Version Specific FAQs

* [v0.7.3.post1 FAQ & Feedback](https://github.com/vllm-project/vllm/issues)
* [v0.9.2rc1 FAQ & Feedback](https://github.com/vllm-project/vllm/issues)

## General FAQs

### 1. What devices are currently supported?

vLLM-GCU currently supports:

* Enflame **S60** Training Card
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

For full compatibility and graph-mode acceleration, see [Model Support Matrix (GCU)](https://github.com/enflame-tech/vllm-gcu/blob/main/docs/model_support.md).

---

### 4. What features does vLLM-GCU support?

* Graph-mode execution on GCU
* OpenAI-compatible REST API
* FP16 / BF16 precision
* 4-bit and 8-bit quantization
* KV Cache for efficient decoding
* Tensor parallelism and pipeline parallelism on multi-card setups
* Static / dynamic batch (continuous batching)
* Offline batch inference
* Int8 KV Cache
* Chunked Prefill

---

### 5. How is the performance on GCU?

vLLM-GCU accelerates models like `Qwen2.5`, `Qwen3`, `LLaMa3`, `DeepSeek`, and `InternLM` by using graph-mode execution. You can further optimize with:

* Enable `--enable-graph-mode`
* Using parallel inference by setting `--tensor-parallel-size`

---

### 6. How does vLLM-GCU integrate with vLLM?

It is implemented as a **plugin backend** to vLLM using Enflame’s GCU kernel interface. Make sure vllm-gcu and vllm use **matching versions** (e.g. `vllm==0.9.1`, `vllm-gcu==0.9.1`).

---

### 7. Does vLLM-GCU support quantization?

Yes. vLLM-GCU supports **w8a8 (GPTQ, 8-bit)** and **w4a16 (AWQ 4-bit)** quantization using GCU-optimized kernels.

---

### 8. How to run AWQ (w4a16) DeepSeek or Qwen models?

Follow the model tutorial and add `--quantization w4a16` to `vllm serve` or `LLM()` initialization.

---

### 9. How to avoid Out Of Memory (OOM) on GCU?

* Reduce `--gpu-memory-utilization`, default is `0.9`
* Enable Enflame memory optimizations:

```bash
export TOPS_ENABLE_VMEM=1
export TOPS_MEM_POOL_ENABLE=1
```

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

Use greedy decoding and set the following envs:

```bash
export GCU_DETERMINISTIC=1
export TOPS_DISABLE_LCOC=1
```

And set temperature to zero:

```python
SamplingParams(temperature=0)
```

---

### 12. How to contact the GCU community?

* [GitHub Issues](https://github.com/enflame-tech/vllm-gcu/issues)
* Join our **WeChat dev group** (QR code in the repo)
* Follow **weekly office hours** via Tencent Meeting (announced on GitHub)
* Discuss in [vLLM Forum GCU Channel](https://discuss.vllm.ai)

---
