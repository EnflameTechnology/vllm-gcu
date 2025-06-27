## internlm

### internlm-7b

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url:[internlm-7b](https://huggingface.co/internlm/internlm-7b/)

*  branch:`main`

*  commit id:`154a736f18ae86c93339e345c145d65d03696156`

将上述url设定的路径下的内容全部下载到`internlm-7b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm-7b] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm-7b] \
 --tokenizer=[path of internlm-7b] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=2048 \
 --dtype=float16 \
 --enforce-eager
```

注：
*  本模型支持的`max-model-len`为2048；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm-20b

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url:[internlm-20b](https://huggingface.co/internlm/internlm-20b)

*  branch:`main`

*  commit id:`2d83118d863d24565da1f9c6c0fe99d3e882f25c`

将上述url设定的路径下的内容全部下载到`internlm-20b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm-chat-20b] \
 --tensor-parallel-size=2 \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm-20b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of internlm-20b] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=32 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm-chat-20b

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url:[internlm-chat-20b](https://huggingface.co/internlm/internlm-chat-20b/tree/main)

*  branch:`main`

*  commit id:`ce6e0150bc4b525c44f6e450569a05705dcb4e72`

将上述url设定的路径下的内容全部下载到`internlm-chat-20b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm-chat-20b] \
 --tensor-parallel-size=2 \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm-chat-20b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of internlm-20b] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=32 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency；

### internlm-7b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`InternLM-7b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`internlm_7b_w8a16_gptq`文件夹中。
- `internlm_7b_w8a16_gptq`目录结构如下所示：

```shell
internlm_7b_w8a16_gptq/
    ├── config.json
    ├── configuration_internlm.py
    ├── generation_config.json
    ├── index.html.tmp
    ├── modeling_internlm.py
    ├── model.safetensors
    ├── pytorch_model.bin.index.json
    ├── quantize_config.json
    ├── special_tokens_map.json
    ├── tokenization_internlm.py
    ├── tokenizer_config.json
    ├── tokenizer.model
    └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm-7b-w8a16_gptq] \
 --demo=te \
 --output-len=256 \
 --dtype=float16 \
 --quantization gptq
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm-7b-w8a16_gptq] \
 --max-model-len=2048 \
 --tokenizer=[path of internlm-7b-w8a16_gptq] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --enforce-eager
```

注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm2_5-7b-chat

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url:[internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat/tree/main)

*  branch:`main`

*  commit id:`9b8d9553846ecf6393f3408fa9d3ec9928fdab4d`

将上述url设定的路径下的内容全部下载到`internlm2_5-7b-chat`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2_5-7b-chat] \
 --demo=tc \
 --dtype=bfloat16 \
 --output-len=128
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2_5-7b-chat] \
 --tokenizer=[path of internlm2_5-7b-chat] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --trust-remote-code
```

注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### internlm2_5-20b-chat

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url:[internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)

*  branch:`main`

*  commit id:`3a276a1dedc6863be72505a6a721c7c59d0f818c`

将上述url设定的路径下的内容全部下载到`internlm2_5-20b-chat`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2_5-20b-chat] \
 --demo=tc \
 --tensor-parallel-size=2 \
 --dtype=bfloat16 \
 --output-len=128 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2_5-20b-chat] \
 --tokenizer=[path of internlm2_5-20b-chat] \
 --tensor-parallel-size=2 \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16 \
 --max-model-len=32768
```

注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm2-7b

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url:[internlm2-7b](https://huggingface.co/internlm/internlm2-7b)

*  branch:`main`

*  commit id:`530fc706c606b1af1145c662877a7d99ad79d623`

将上述url设定的路径下的内容全部下载到`internlm2-7b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2-7b] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2-7b] \
 --tokenizer=[path of internlm2-7b] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --dtype=float16 \
 --enforce-eager
```

注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm2-20b

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url:[internlm2-20b](https://huggingface.co/internlm/internlm2-20b)

*  branch:`main`

*  commit id:`f363ea8a116b3ea829c7a068ca24bc9d3e668083`

将上述url设定的路径下的内容全部下载到`internlm2-20b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2-20b] \
 --tensor-parallel-size=2 \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2-20b] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --tokenizer=[path of internlm2-20b] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### internlm2-chat-20b

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url:[internlm2-chat-20b](https://huggingface.co/internlm/internlm2-chat-20b)

*  branch:`main`

*  commit id:`e013aec8d021d3cd81bd11f12cb5176b77bb8e6a`

将上述url设定的路径下的内容全部下载到`internlm2-chat-20b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2-chat-20b] \
 --tensor-parallel-size=2 \
 --demo=tc \
 --dtype=float16 \
 --max-model-len=32768 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2-chat-20b] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --tokenizer=[path of internlm2-chat-20b] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

