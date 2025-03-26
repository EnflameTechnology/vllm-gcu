## internlm

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



### internlm2_5-7b-chat-w4a16-gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`internlm2_5-7b-chat-w4a16-gptq-groupsize64.tar`文件并解压，将压缩包内的内容全部拷贝到`internlm2_5-7b-chat-w4a16-gptq`文件夹中。
- `internlm2_5-7b-chat-w4a16-gptq`目录结构如下所示：

```shell
internlm2_5-7b-chat-w4a16-gptq/
    ├── config.json
    ├── configuration_internlm2.py
    ├── generation_config.json
    ├── modeling_internlm2.py
    ├── model.safetensors
    ├── quantize_config.json
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenization_internlm2_fast.py
    ├── tokenization_internlm2.py
    ├── tokenizer_config.json
    ├── tokenizer.model
    └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=128 \
 --device gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --trust-remote-code \
 --tensor-parallel-size=1 \
 --tokenizer=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --device gcu
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
