# Quantization Guide

Model quantization is a technique to reduce model size and memory usage by lowering the precision of weights and activations—enabling faster inference and deployment on resource-constrained environments.

Since `v0.9.0rc2`, quantization is **experimentally supported** in vLLM. This guide covers supported methods, model preparation, and usage examples for quantized models.

> ✅ Currently, **LLaMa**, **Mistral**, **Qwen** and **DeepSeek** models are well-tested. More models and formats will be supported in future versions.

---

## Supported Methods

| Method         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `GPTQ`         | 4-bit group quantization. Group size must be 64 or its multiple |
| `AWQ`          | Activation-aware weight quantization. Group size = 64 required  |
| `W8A16`        | Weights in INT8, activations in FP16                            |
| `INT8 KVCache` | KV cache in INT8 format. Requires separate configuration        |

> ❌ **Not supported yet**: `g_idx` shuffle (GPTQ), SqueezeLLM, FP8, `gptq_marlin`, etc.

---

## Install TopsCompressor

vLLM-GCU uses [TopsCompressor], a model compression tool optimized for the GCU platform.

> ⚠️ Only the following tag is **compatible**:

```bash
git clone https://gitee.com/enflame-tech/topscompressor
cd topscompressor
python setup.py install
```

---

## Quantize a Model

Using [DeepSeek-V2-Lite](https://modelscope.cn/models/deepseek-ai/DeepSeek-V2-Lite) as an example:

```bash
cd example/DeepSeek
python3 quant_deepseek.py \
  --model_path {original_model_path} \
  --save_directory {quantized_model_save_path} \
  --device_type cpu \
  --act_method 2 \
  --w_bit 8 \
  --a_bit 8 \
  --is_dynamic True
```

📌 You can also download pre-quantized models from [ModelScope](https://www.modelscope.cn/models/vllm-gcu/DeepSeek-V2-Lite-W8A8), for testing purposes.

---

## Output Files

After quantization, you’ll see files like:

```bash
.
├── config.json                       # No "quantization_config" field
├── quant_model_description.json     # Metadata for quantized weights
├── quant_model_weight_w8a8_dynamic-*.safetensors
├── configuration_deepseek.py
├── tokenization_deepseek_fast.py
├── tokenizer_config.json
└── ...
```

Ensure `config.json` does **not** contain a `quantization_config` field.

---

## Running Quantized Models

### Offline Inference

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The future of AI is"]
params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(
    model="{quantized_model_save_path}",
    max_model_len=2048,
    trust_remote_code=True,
    quantization="w4a16"
)

outputs = llm.generate(prompts, params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### Online Inference

```bash
vllm serve {quantized_model_save_path} \
  --served-model-name "deepseek-v2-lite-w8a8" \
  --max-model-len 2048 \
  --quantization w4a16 \
  --trust-remote-code
```

---

## FAQs

### Q1: `KeyError: 'xxx.layers.0.self_attn.q_proj.weight'`

* Ensure `quantization="gcu"` is specified.
* Confirm model was quantized with `modelslim-VLLM-8.1.RC1.b020_001`.
* If issue persists, submit a GitHub issue—some models may require adaptation.

---

### Q2: `Could not locate configuration_deepseek.py`

* This error is fixed in the correct ModelSlim version. Re-quantize with the compatible tag.
