# Graph Mode Guide

```{note}
This feature is currently **experimental**. Future versions may introduce behavioral changes to configuration, model coverage, or performance characteristics.
```

vLLM-GCU supports graph execution to improve performance for certain models. This guide explains how to use graph mode. Graph mode is only available with the **V1 Engine**.

> ✅ Graph mode is **enabled by default** for supported models on V1 Engine.

## Getting Started

From `v0.9.1`, vLLM enables graph mode automatically for supported models when using the V1 engine. If you encounter issues, consider:

* Falling back to eager mode by setting `enforce_eager=True`.
* Submitting a GitHub issue for support.

---

## Using TopGraph (Default)

No extra configuration is needed for TopGraph. Simply load a supported model (e.g., Qwen2.5) with the V1 Engine:

**Offline example:**

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2-7B-Instruct")
outputs = llm.generate("Hello, how are you?")
```

**Online example:**

```bash
vllm serve Qwen/Qwen2-7B-Instruct
```

---

## Fallback to Eager Mode

If graph mode causes issues, you can disable it by forcing **eager mode**:

**Offline example:**

```python
from vllm import LLM

llm = LLM(model="your_model", enforce_eager=True)
outputs = llm.generate("Hello, how are you?")
```

**Online example:**

```bash
vllm serve your_model --enforce-eager
```
