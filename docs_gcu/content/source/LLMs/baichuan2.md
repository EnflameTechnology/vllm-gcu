## baichuan2

### baichuan2-7b
#### 模型下载
*  url:[baichuan2-7b](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/)

*  branch:`main`

*  commit id:`364ead367078c68c8deef6a319053302b330aa1f`

将上述url设定的路径下的内容全部下载到`baichuan2-7B-base`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-7B-base] \
 --demo=tc \
 --output-len=20 \
 --dtype=float16 \
 --gpu-memory-utilization=0.945
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-7B-base] \
 --tokenizer=[path of baichuan2-7B-base] \
 --input-len=512 \
 --output-len=240 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --gpu-memory-utilization=0.945 \
 --enforce-eager
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


#### 基于OpenCompass进行mmlu数据集评测

1. 安装OpenCompass

执行 [OpenCompass的安装步骤](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md#%EF%B8%8F-%E5%AE%89%E8%A3%85)

注：建议使用OpenCompass0.3.1版本。如果安装依赖时安装了和torch_gcu不一致的版本，请重新手动安装。

注：需要安装以下依赖：

```shell
python3 -m pip install opencv-python==4.9.0.80
python3 -m pip install huggingface-hub==0.25.2
# for x86_64
python3 -m pip install torchvision==0.21.0+cpu -i https://download.pytorch.org/whl/cpu
# for aarch64
python3 -m pip install torchvision==0.21.0
# for x86_64 and python_version>=3.10
python3 -m pip install importlib-metadata==8.5.0
# for aarch64 and python_version>=3.10
python3 -m pip install importlib-metadata==4.6.4
```

2. 准备config文件

将下面的配置信息存为一个python文件，放入OpenCompass中如下路径`configs/models/baichuan/vllm_baichuan2_7b_base.py`

```python
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='baichuan2-7b-base-vllm',
        path='path/to/baichuan2-7B-base',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', enforce_eager=True)
    )
]


```

执行以下命令

```
python3 run.py \
 --models vllm_baichuan2_7b_base \
 --datasets mmlu_gen
```


### baichuan2-13b
#### 模型下载
*  url:[baichuan2-13b](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)

*  branch:`main`

*  commit id:`c6f590cab590cf33e78ad834dbd5f9bd6df34a94`

将上述url设定的路径下的内容全部下载到`baichuan2-13B-base`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-13B-base] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-13B-base] \
 --tokenizer=[path of baichuan2-13B-base] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

### baichuan2-7B-base-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Baichuan2-7b-base-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`baichuan2-7B-base-w8a16_gptq`文件夹中。
- `baichuan2-7B-base-w8a16_gptq`目录结构如下所示：

```shell
baichuan2-7B-base-w8a16_gptq/
├── config.json
├── configuration_baichuan.py
├── generation_utils.py
├── modeling_baichuan.py
├── model.safetensors
├── pytorch_model.bin.index.json
├── quantize_config.json
├── quantizer.py
├── special_tokens_map.json
├── tokenization_baichuan.py
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-7B-base-w8a16_gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization gptq \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-7B-base-w8a16_gptq] \
 --tokenizer=[path of baichuan2-7B-base-w8a16_gptq] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --enforce-eager
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-13B-base-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Baichuan2-13b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`baichuan2-13B-base-w8a16_gptq`文件夹中。
- `baichuan2-13B-base-w8a16_gptq`目录结构如下所示：

```shell
baichuan2-13B-base-w8a16_gptq/
├── config.json
├── configuration_baichuan.py
├── generation_utils.py
├── modeling_baichuan.py
├── model.safetensors
├── pytorch_model.bin.index.json
├── quantize_config.json
├── quantizer.py
├── special_tokens_map.json
├── tokenization_baichuan.py
├── tokenizer_config.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-13B-base-w8a16_gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization gptq \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-13B-base-w8a16_gptq] \
 --max-model-len=4096 \
 --tokenizer=[path of baichuan2-13B-base-w8a16_gptq] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-13B-w4a16-awq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Baichuan2-13b_W4A16_AWQ.tar`文件并解压，将压缩包内的内容全部拷贝到`baichuan2-13B-w4a16-awq`文件夹中。
- `baichuan2-13B-w4a16-awq`目录结构如下所示：

```shell
baichuan2-13B-w4a16-awq/
├── config.json
├── configuration_baichuan.py
├── generation_config.json
├── generation_utils.py
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── modeling_baichuan.py
├── model.safetensors.index.json
├── quantizer.py
├── special_tokens_map.json
├── tokenization_baichuan.py
├── tokenizer_config.json
└── tokenizer.model
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-13B-w4a16-awq] \
 --demo=tc \
 --dtype=float16 \
 --quantization awq \
 --output-len=100
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-13B-w4a16-awq] \
 --max-model-len=4096 \
 --tokenizer=[path of baichuan2-13B-w4a16-awq] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=awq
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-13B_w8a8c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`baichuan2-13b_w8a8c8.tar`文件并解压，将压缩包内的内容全部拷贝到`baichuan2-13B_w8a8c8'文件夹中。
- `baichuan2-13B_w8a8c8`目录结构如下所示：

```shell
baichuan2-13B_w8a8c8/
── config.json
├── configuration_baichuan.py
├── generation_utils.py
├── int8_kv_cache_vllm.json
├── modeling_baichuan.py
├── model.safetensors
├── quantize_config.json
├── quantizer.py
├── tokenization_baichuan.py
├── tokenizer_config.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-13B_w8a8c8] \
 --demo=tc \
 --dtype=float16 \
 --output-len=100 \
 --quantization-param-path=[path of baichuan2-13B_w8a8c8/int8_kv_cache.json] \
 --kv-cache-dtype=int8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-13B_w8a8c8] \
 --max-model-len=4096 \
 --tokenizer=[path of baichuan2-13B_w8a8c8] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --quantization-param-path=[path of baichuan2-13B_w8a8c8/int8_kv_cache.json] \
 --kv-cache-dtype=int8
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`baichuan2-13b_w8a8c8.tar`文件并解压，将压缩包内的内容全部拷贝到`baichuan2-13B_w8a8c8'文件夹中。
- `baichuan2-13B_w8a8c8`目录结构如下所示：

```shell
baichuan2-13B_w8a8c8/
── config.json
├── configuration_baichuan.py
├── generation_utils.py
├── int8_kv_cache_vllm.json
├── modeling_baichuan.py
├── model.safetensors
├── quantize_config.json
├── quantizer.py
├── tokenization_baichuan.py
├── tokenizer_config.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of baichuan2-13B_w8a8c8] \
 --demo=tc \
 --dtype=float16 \
 --output-len=100 \
 --quantization-param-path=[path of baichuan2-13B_w8a8c8/int8_kv_cache.json] \
 --kv-cache-dtype=int8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of baichuan2-13B_w8a8c8] \
 --max-model-len=4096 \
 --tokenizer=[path of baichuan2-13B_w8a8c8] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --quantization-param-path=[path of baichuan2-13B_w8a8c8/int8_kv_cache.json] \
 --kv-cache-dtype=int8
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
