# Quantization Guide

Model quantization is a technique to reduce model size and memory usage by lowering the precision of weights and activations—enabling faster inference and deployment on resource-constrained environments.

Since `v0.8.0`, quantization is **experimentally supported** in vLLM-GCU. This guide covers supported methods, model preparation, and usage examples for quantized models.

> ✅ Currently, **LLaMa**, **Mistral**, **Qwen**, **GLM4** and **DeepSeek** models are well-tested. More models and formats will be supported in future versions.

---

## Supported Methods

| Method         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `GPTQ`         | 4-bit group quantization. Group size must be 64 or its multiple |
| `AWQ`          | Activation-aware weight quantization. Group size = 64 required  |
| `W8A16`        | Weights in INT8, activations in FP16                            |

> ❌ **Not supported yet**: `g_idx` shuffle (GPTQ), `FP8`, `marlin`, etc.

---

## Install TopsCompressor

vLLM-GCU uses [TopsCompressor], a model compression tool optimized for the GCU platform.

> ⚠️ `TopsCompressor` can be selectedly installed during installation of `TopsRider`.

Or install independently with:

```bash
pip insall topscompressor-<version>-py3.10-none-any
```

---

## Quantize a Model

> Quantize models with Enflame custom tool (`TopsCompressor`) to achieve optimal accuracy performance on Enflame GCU devices.

Using [Meta-Llama-3.1-8B-Instruct](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct) as an example:

Save the following code as `basic_quant.py`

:::::{tab-set}
::::{tab-item} Basic Quant

```{code-block} python
   :substitutions:
import argparse
import torch
from topscompressor.quantization.quantize import quantize, save_quantized_model
from topscompressor.quantization.config import QuantConfig
 
 
def main(args):
    dtype = {
        'auto': 'auto',
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32
    }
    quant_config = QuantConfig.create_config('w8a8', quant_scheme=args.quant_scheme)
    calib_data_name = 'wikitext'
    calib_data_config = {
        'name': '',
        'split': 'validation',
    }
    model = quantize(
        args.model_name_or_path,
        quant_config,
        calib_data_name,
        calib_data_load_fn_kwargs=calib_data_config,
        calib_data_max_len=512,
        n_samples=args.nsamples,
        custom_model_fn=args.custom_model_fn,
        device=args.device,
        torch_dtype=dtype[args.dtype]
    )
    save_quantized_model(model, quant_config, args.save_dir)
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="quantize model sample of w8a8-int8.")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=True,)
    parser.add_argument("--save_dir",
                        type=str,
                        required=True,)
    parser.add_argument("--quant_scheme",
                        type=str,
                        choices=['w_channel_act_token',
                                 'w_channel_act_static_tensor'],
                        default='w_channel_act_token')
    parser.add_argument("--device",
                        type=str,
                        choices=['gcu', 'cuda'],
                        default='gcu',)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--custom_model_fn",
                        type=str,
                        help="custom model config file(*.json) or python file(*.py)",
                        default=None)
    parser.add_argument("--dtype",
                        type=str,
                        help="model dtype",
                        default='auto',
                        choices=['auto', 'float16', 'bfloat16', 'float32'])
    args = parser.parse_args()
    main(args)
```
::::
:::::


Use `basic_quant.py` to quantize models, e.g., `Meta-Llama-3.1-8B-Instruct`.

```bash
python basic_quant.py \
--model_name_or_path=/data/Meta-Llama-3.1-8B-Instruct \
--save_dir=/data/Meta-Llama-3.1-8B-Instruct_W8A8 \
--quant_scheme=w_channel_act_static_tensor \
--device=gcu \
--nsamples=128 \
--dtype=bfloat16
```

📌 Alternatively, you can also download pre-quantized models from [ModelScope](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4) for testing purposes.

---

## Running Quantized Models

### Offline Inference

```python
from vllm import LLM, SamplingParams

prompts = ["Hello, my name is", "The future of AI is"]
params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(
    model="/data/Meta-Llama-3.1-8B-Instruct_W8A8",
    max_model_len=4096,
    tensor-parallel-size=2
    trust_remote_code=True,
    quantization="w8a8"
)

outputs = llm.generate(prompts, params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")
```

### Online Inference

```bash
vllm serve /data/Meta-Llama-3.1-8B-Instruct_W8A8 \
  --served-model-name "Meta-Llama-3.1-8B-Instruct_W8A8" \
  --max-model-len 4096 \
  --quantization w8a8 \
  --tensor-parallel-size 2 \
  --trust-remote-code
```
