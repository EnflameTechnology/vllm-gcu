## Mistral

### Mistral-7B-v0.1

#### 模型下载
*  url: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/tree/main)

*  branch: `main`

*  commit id: `26bca36`

将上述url设定的路径下的内容全部下载到`mistral-7b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mistral-7b] \
 --demo=te \
 --dtype=float16 \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mistral-7b] \
 --max-model-len=32768 \
 --tokenizer=[path of mistral-7b] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### mixtral-8x7B-v0.1
#### 模型下载
*  url:[mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/tree/main)

*  branch:`main`

*  commit id:`985aa05`

将上述url设定的路径下的内容全部下载到`mixtral-8x7B-v0.1`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mixtral-8x7B-v0.1] \
 --tensor-parallel-size=4 \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mixtral-8x7B-v0.1] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --tokenizer=[path of mixtral-8x7B-v0.1] \
 --input-len=100 \
 --output-len=180 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Mixtral-8x22B-v0.1

本模型推理及性能测试需要8张enflame gcu。
#### 模型下载
* url:[mixtral-8x22b-v0.1](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/tree/main)
* branch: `main`
* commit id: `ab1e8c1950cf359e2a25de9b274ab836adb6dbab`

将上述url设定的路径下的内容全部下载到`mixtral-8x22b-v0.1`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mixtral_8x22b] \
 --tensor-parallel-size=8 \
 --output-len=512 \
 --demo=te \
 --dtype=bfloat16 \
 --device=gcu \
 --max-model-len=8192
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mixtral_8x22b] \
 --device=gcu \
 --max-model-len=8192 \
 --tokenizer=[path of mixtral_8x22b] \
 --input-len=1024 \
 --output-len=3072 \
 --num-prompts=1 \
 --tensor-parallel-size=8 \
 --block-size=64
```
注：
*  本模型支持的`max-model-len`为65536；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### mixtral-8x7B-v0.1-w8a16_gptq

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`mixtral-8x7B-v0.1-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`mixtral-8x7B-v0.1-w8a16_gptq`文件夹中。
- `mixtral-8x7B-v0.1-w8a16_gptq`目录结构如下所示：

```shell
mixtral-8x7B-v0.1-w8a16_gptq/
├── config.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json

```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mixtral-8x7B-v0.1-w8a16_gptq] \
 --tensor-parallel-size=4 \
 --demo=te \
 --dtype=float16 \
 --output-len=64 \
 --quantization gptq \
 --max-model-len=4096
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mixtral-8x7B-v0.1-w8a16_gptq] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --tokenizer=[path of mixtral-8x7B-v0.1-w8a16_gptq] \
 --input-len=100 \
 --output-len=180 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Mixtral-8x22B-v0.1-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`mixtral-8x22B-v0.1-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`mixtral-8x22B-v0.1-w8a16_gptq`文件夹中。
- `mixtral-8x22B-v0.1-w8a16_gptq`目录结构如下所示：

```shell
mixtral-8x22B-v0.1-w8a16_gptq/
├── config.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json

```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mixtral-8x22B-v0.1-w8a16_gptq] \
 --tensor-parallel-size=8 \
 --max-model-len=65536 \
 --output-len=512 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization gptq
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mixtral-8x22B-v0.1-w8a16_gptq] \
 --max-model-len=65536 \
 --tokenizer=[path of mixtral-8x22B-v0.1-w8a16_gptq] \
 --input-len=8192 \
 --output-len=8192 \
 --num-prompts=1 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --device=gcu \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为65536；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### mixtral-8x7B-gptq

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`mixtral-8x7B-gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`mixtral-8x7B-gptq`文件夹中。
- `mixtral-8x7B-gptq`目录结构如下所示：

```shell
mixtral-8x7B-gptq
├── config.json
├── generation_config.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of mixtral-8x7B-gptq] \
 --tensor-parallel-size=4 \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --quantization gptq \
 --block-size=64 \
 --num-prompts=1 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of mixtral-8x7B-gptq] \
 --tensor-parallel-size=4 \
 --max-model-len=8192 \
 --tokenizer=[path of mixtral-8x7B-gptq] \
 --input-len=4096 \
 --output-len=4096 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Mixtral-8x22B-Instruct-v0.1

本模型推理及性能测试需要8张enflame gcu。
#### 模型下载
* url:[Mixtral-8x22B-Instruct-v0.1](https://www.modelscope.cn/models/AI-ModelScope/Mixtral-8x22B-Instruct-v0.1/files)
* branch: `master`
* commit id: `eb269184`

将上述url设定的路径下的内容全部下载到`Mixtral-8x22B-Instruct-v0.1`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Mixtral-8x22B-Instruct-v0.1] \
 --tensor-parallel-size=8 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device=gcu \
 --max-model-len=32768 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Mixtral-8x22B-Instruct-v0.1] \
 --device=gcu \
 --max-model-len=32768 \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为65536；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;