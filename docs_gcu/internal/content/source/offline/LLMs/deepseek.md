## deepseek

### deepseek-moe-16b-base
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [deepseek-moe-16b-base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main)

*  branch: `main`

*  commit id: `521d2bc`

将上述url设定的路径下的内容全部下载到`deepseek-moe-16b-base`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-moe-16b-base] \
 --demo=te \
 --output-len=256 \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-moe-16b-base] \
 --tokenizer=[path of deepseek-moe-16b-base] \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### deepseek-llm-67b-base
本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [deepseek-llm-67b-base](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/tree/main)

*  branch: `main`

*  commit id: `c3f813a`

将上述url设定的路径下的内容全部下载到`deepseek-llm-67b-base`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-llm-67b-base] \
 --tensor-parallel-size=4 \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-llm-67b-base] \
 --tokenizer=[path of deepseek-llm-67b-base] \
 --tensor-parallel-size=4 \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### deepseek-llm-67b-chat
本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [deepseek-llm-67b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-67b-chat/tree/main)

*  branch: `main`

*  commit id: `79648be`

将上述url设定的路径下的内容全部下载到`deepseek-llm-67b-chat`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-llm-67b-chat] \
 --tensor-parallel-size=4 \
 --dtype=float16 \
 --demo=ch \
 --output-len=20 \
 --template=default
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-llm-67b-chat] \
 --tokenizer=[path of deepseek-llm-67b-chat] \
 --tensor-parallel-size=4 \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### deepseek-moe-16b-base-w4a16
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`deepseek-moe-16b-base-w4a16.tar`文件以及并解压，将压缩包内的内容全部拷贝到`deepseek-moe-16b-base-w4a16`文件夹中。
- `deepseek-moe-16b-base-w4a16`目录结构如下所示：

```shell
deepseek-moe-16b-base-w4a16/
  ├──config.json
  ├──configuration_deepseek.py
  ├──generation_config.json
  ├──modeling_deepseek.py
  ├──model.safetensors
  ├──quantize_config.json
  ├──tokenizer_config.json
  ├──tokenizer.json
  ├──tops_quantize_info.json
```

将上述url设定的路径下的内容全部下载到`deepseek-moe-16b-base-w4a16`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-moe-16b-base-w4a16] \
 --demo=te \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --output-len=128 \
 --device gcu \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code \
 --quantization gptq \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-moe-16b-base-w4a16] \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --quantization gptq \
 --dtype=float16 \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### deepseek-moe-16b-chat
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [deepseek-moe-16b-chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat/tree/main)

*  branch: `main`

*  commit id: `eefd8ac`

将上述url设定的路径下的内容全部下载到`deepseek-moe-16b-chat`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-moe-16b-chat] \
 --demo=ch \
 --output-len=256 \
 --dtype=float16 \
 --template=default
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-moe-16b-chat] \
 --tokenizer=[path of deepseek-moe-16b-chat] \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`--num-prompts=16`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### deepseek-coder-6.7b-base
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base/tree/main)

*  branch: `main`

*  commit id: `ce2207a`

将上述url设定的路径下的内容全部下载到`deepseek-coder-6.7b-base`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-coder-6.7b-base] \
 --demo=cc \
 --output-len=256 \
 --dtype=float16 \
 --max-model-len=1024
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-coder-6.7b-base] \
 --tokenizer=[path of deepseek-coder-6.7b-base] \
 --max-model-len=4096 \
 --input-len=1500 \
 --output-len=256 \
 --num-prompts=2 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;