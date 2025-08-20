# Graph Mode Guide

```{note}
This feature is currently **experimental**. Future versions may introduce behavioral changes to configuration, model coverage, or performance characteristics.
```

vLLM-GCU supports graph execution to improve performance for certain models. This guide explains how to use graph mode.

> ✅ Graph mode is **enabled by default** for supported models.

## Getting Started

From `v0.8.0`, vLLM-GCU enables graph mode automatically for supported models. If you encounter issues, consider:

* Falling back to eager mode by setting `enforce_eager=True`.
* Submitting a GitHub issue for support.

---

## Using GcuGraph (Default)

No extra configuration is needed for GcuGraph. Simply load a supported model (e.g., Qwen2.5):

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
