## deepseek

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
 --gpu-memory-utilization 0.945 \
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
 --gpu-memory-utilization 0.945 \
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

### DeepSeek-V2-Lite-Chat
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [DeepSeek-V2-Lite-Chat](https://www.modelscope.cn/models/deepseek-ai/deepseek-v2-lite-chat/files)

*  branch: `master`

*  commit id: `d174ca84`

将上述url设定的路径下的内容全部下载到`deepseek-v2-lite-chat`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of DeepSeek-V2-Lite-Chat] \
 --tensor-parallel-size=1 \
 --max-model-len=8192 \
 --output-len=512 \
 --demo=dch \
 --dtype=bfloat16 \
 --template=default \
 --device=gcu \
 --trust-remote-code
```

### deepseek-moe-16b-base-w8a8c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`deepseek-moe-16b-base-w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`deepseek-moe-16b-base-w8a8`文件夹中。
- deepseek-moe-16b-base_int8_kv_cache.json文件位于`deepseek-moe-16b-base-w8a8`文件夹中。
- `deepseek-moe-16b-base-w8a8`目录结构如下所示：

```shell
deepseek-moe-16b-base-w8a8
├── config.json
├── configuration_deepseek.py
├── deepseek-moe-16b-base_int8_kv_cache.json
├── generation_config.json
├── modeling_deepseek.py
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
└── tops_quantize_info.json
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
  --demo='te' \
  --model=[path of deepseek-moe-16b-base-w8a8] \
  --quantization-param-path [path of deepseek-moe-16b-base_int8_kv_cache.json] \
  --kv-cache-dtype int8 \
  --num-prompts 1 \
  --max-model-len=4096 \
  --output-len=128 \
  --device=gcu \
  --dtype=float16 \
  --quantization=w8a8 \
  --gpu-memory-utilization=0.945 \
  --tensor-parallel-size=1
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-moe-16b-base-w8a8] \
 --quantization-param-path [path of deepseek-moe-16b-base_int8_kv_cache.json] \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --kv-cache-dtype int8 \
 --device gcu \
 --gpu-memory-utilization 0.945
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Qwen-1.5B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-1.5B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

*  branch: `master`

*  commit id: `b7993c1b`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-1.5B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Qwen-1.5B] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Qwen-1.5B] \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Qwen-1.5B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Qwen-7B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

*  branch: `master`

*  commit id: `6bf9b8f2`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-7B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Qwen-14B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-14B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)

*  branch: `master`

*  commit id: `3b1bf094`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-14B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Qwen-14B] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Qwen-14B] \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Qwen-14B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Qwen-32B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-32B](https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)

*  branch: `master`

*  commit id: `98e5cdec`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-32B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Qwen-32B] \
 --tensor-parallel-size 4 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Qwen-32B] \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Qwen-32B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Llama-8B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Llama-8B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

*  branch: `master`

*  commit id: `b1a59cb3`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Llama-8B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Llama-8B] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Llama-8B] \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Llama-8B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-Distill-Llama-70B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)

*  branch: `master`

*  commit id: `7643071c`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Llama-70B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-Distill-Llama-70B] \
 --tensor-parallel-size 8 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Llama-70B] \
 --tensor-parallel-size 8 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.945 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Llama-70B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### DeepSeek-R1-awq
本模型推理及性能测试需要32张enflame gcu。

#### 模型下载
*  url: [DeepSeek-R1-awq](https://www.modelscope.cn/models/cognitivecomputations/DeepSeek-R1-awq)

*  branch: `master`

*  commit id: `036f5c9a`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-awq`文件夹中。

#### 使用TP+PP的并行方案部署模型

##### 设置环境变量
```shell
apt install net-tools -y
ifconfig -a
# 从结果中选择包含inet字段且内容与机器实际ip一致的字段，填入[interface name]

export ECCL_SOCKET_IFNAME=[interface name]
export TP_SOCKET_IFNAME=[interface name]
export GLOO_SOCKET_IFNAME=[interface name]
```

##### 启动ray集群
```shell
# 选择其中一台服务器作为主节点，主节点ip填入[master node ip]
# 主节点启动命令
ray start --head --port=6379 --num-gpus=8

# 从节点启动命令
ray start --address="[master node ip]:6379" --num-gpus=8
```

##### 批量离线推理
```shell
# 在主节点上启动命令
python3.10 -m vllm_utils.benchmark_test \
 --model [path of DeepSeek-R1-awq] \
 --tensor-parallel-size 8 \
 --pipeline-parallel-size 4 \
 --max-model-len 12288 \
 --output-len 128 \
 --demo te \
 --dtype bfloat16 \
 --device gcu \
 --trust-remote-code \
 --quantization moe_wna16 \
 --distributed_executor_backend ray \
 --async-engine
```

##### 性能测试

```shell
# 在主节点上启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of DeepSeek-R1-awq] \
 --dtype bfloat16 \
 --trust-remote-code \
 --max-model-len=12288 \
 --tensor-parallel-size=8 \
 --pipeline-parallel-size=4 \
 --block-size=64 \
 --quantization moe_wna16 \
 --distributed_executor_backend ray \
 --max-num-batched-tokens 4096 \
 --max-num-seqs 256 \
 --gpu-memory-utilization 0.8 \
 --enable-chunked-prefill


# 在主节点上启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model=[path of DeepSeek-R1-awq] \
 --num-prompts 8 \
 --random-input-len 4096 \
 --random-output-len 256 \
 --trust-remote-code \
 --ignore_eos \
 --keep-special-tokens \
 --strict-in-out-len
```
注：
*  本模型支持的`max-model-len`为163840，在32张S60上最大可支持65536；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

#### 使用EP+DP的并行方案部署模型

##### 性能测试

```shell
apt install net-tools -y
ifconfig -a
# 从结果中选择包含inet字段且内容与机器实际ip一致的字段，填入[interface name]

# EP+DP并行方案需要在4台装有8张S60的服务器上，进行一共8次服务端启动，每次使用4张S60
# 选择其中一台服务器作为主节点，主节点ip填入[master node ip]
# 当前节点ip填入[current node ip]
# 当TOPS_VISIBLE_DEVICES选择'0,1,2,3'时，配置--port为7555，当TOPS_VISIBLE_DEVICES选择'4,5,6,7'时，配置--port为7556
# 启动时按顺序配置VLLM_DP_RANK为0~7，每次启动对应一个独有的VLLM_DP_RANK
ECCL_SHM_DISABLE=1 \
ECCL_ALLTOALLV_MAXSIZE=29491200 \
TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
GLOO_SOCKET_IFNAME=[interface name] \
TOPS_VISIBLE_DEVICES=['0,1,2,3', '4,5,6,7'] \
TOPS_STREAM_SCHEDULE_CREDIT=4 \
VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
VLLM_USE_V1=0 \
VLLM_DP_MASTER_IP=[master node ip] \
VLLM_DP_MASTER_PORT=54999 \
VLLM_DP_SIZE=8 \
VLLM_DP_RANK=[0, 1, 2, 3, 4, 5, 6, 7] \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
 --host [current node ip] \
 --port [7555, 7556] \
 --model=[path of DeepSeek-R1-awq] \
 --dtype bfloat16 \
 --trust-remote-code \
 --quantization moe_wna16_gcu \
 --max-model-len=4096 \
 --tensor-parallel-size=4 \
 --gpu-memory-utilization=0.8 \
 --block-size=64 \
 --enable-expert-parallel \
 --num-scheduler-steps 8 \
 --compilation-config "{\"cudagraph_capture_sizes\": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}" \
 --scheduling-policy priority
```

```shell
# 在主节点上启动DP Router
# 将4台服务器ip分别填入[IP_Node0],[IP_Node1],[IP_Node2],[IP_Node3]
# 主节点ip填入[master node ip]
python3 -m vllm_utils.router \
 --server-urls "http://[IP_Node0]:7555 \
  http://[IP_Node0]:7556 \
  http://[IP_Node1]:7555 \
  http://[IP_Node1]:7556 \
  http://[IP_Node2]:7555 \
  http://[IP_Node2]:7556 \
  http://[IP_Node3]:7555 \
  http://[IP_Node3]:7556" \
 --host [master node ip] \
 --port 8002 \
 --model [path of DeepSeek-R1-awq]
```

```shell
# 在主节点上启动客户端
# 主节点ip填入[master node ip]
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-awq] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 64 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore-eos \
 --base-url http://[master node ip]:8002 \
 --extra-body '{"priority": 1}'
```
注：
*  本模型支持的`max-model-len`为163840，在32张S60上最大可支持65536；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### DeepSeek-V3-awq
本模型推理及性能测试需要32张enflame gcu。

#### 模型下载
*  url: [DeepSeek-V3-awq](https://www.modelscope.cn/models/cognitivecomputations/DeepSeek-V3-awq)

*  branch: `master`

*  commit id: `92c69306`

将上述url设定的路径下的内容全部下载到`DeepSeek-V3-awq`文件夹中。

#### 使用TP+PP的并行方案部署模型

##### 设置环境变量
```shell
apt install net-tools -y
ifconfig -a
# 从结果中选择包含inet字段且内容与机器实际ip一致的字段，填入[interface name]

export ECCL_SOCKET_IFNAME=[interface name]
export TP_SOCKET_IFNAME=[interface name]
export GLOO_SOCKET_IFNAME=[interface name]
```

##### 启动ray集群
```shell
# 选择其中一台服务器作为主节点，主节点ip填入[master node ip]
# 主节点启动命令
ray start --head --port=6379 --num-gpus=8

# 从节点启动命令
ray start --address="[master node ip]:6379" --num-gpus=8
```

##### 性能测试

```shell
# 在主节点上启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of DeepSeek-V3-awq] \
 --dtype bfloat16 \
 --trust-remote-code \
 --max-model-len=4096 \
 --tensor-parallel-size=8 \
 --pipeline-parallel-size=4 \
 --gpu-memory-utilization=0.9 \
 --block-size=64 \
 --max_num_seqs=256 \
 --quantization=moe_wna16_gcu \
 --distributed_executor_backend=ray


# 在主节点上启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model=[path of DeepSeek-V3-awq] \
 --num-prompts 64 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore_eos \
 --keep-special-tokens \
 --strict-in-out-len
```
注：
*  本模型支持的`max-model-len`为163840，在32张S60上最大可支持65536；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

#### 使用EP+DP的并行方案部署模型

##### 性能测试

```shell
apt install net-tools -y
ifconfig -a
# 从结果中选择包含inet字段且内容与机器实际ip一致的字段，填入[interface name]

# EP+DP并行方案需要在4台装有8张S60的服务器上，进行一共8次服务端启动，每次使用4张S60
# 选择其中一台服务器作为主节点，主节点ip填入[master node ip]
# 当前节点ip填入[current node ip]
# 当TOPS_VISIBLE_DEVICES选择'0,1,2,3'时，配置--port为7555，当TOPS_VISIBLE_DEVICES选择'4,5,6,7'时，配置--port为7556
# 启动时按顺序配置VLLM_DP_RANK为0~7，每次启动对应一个独有的VLLM_DP_RANK
ECCL_SHM_DISABLE=1 \
ECCL_ALLTOALLV_MAXSIZE=29491200 \
TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
GLOO_SOCKET_IFNAME=[interface name] \
TOPS_VISIBLE_DEVICES=['0,1,2,3', '4,5,6,7'] \
TOPS_STREAM_SCHEDULE_CREDIT=4 \
VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
VLLM_USE_V1=0 \
VLLM_DP_MASTER_IP=[master node ip] \
VLLM_DP_MASTER_PORT=54999 \
VLLM_DP_SIZE=8 \
VLLM_DP_RANK=[0, 1, 2, 3, 4, 5, 6, 7] \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python3 -m vllm.entrypoints.openai.api_server \
 --host [current node ip] \
 --port [7555, 7556] \
 --model=[path of deepseek-v3-awq] \
 --dtype bfloat16 \
 --trust-remote-code \
 --quantization moe_wna16_gcu \
 --max-model-len=4096 \
 --tensor-parallel-size=4 \
 --gpu-memory-utilization=0.8 \
 --block-size=64 \
 --enable-expert-parallel \
 --num-scheduler-steps 8 \
 --compilation-config "{\"cudagraph_capture_sizes\": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}" \
 --scheduling-policy priority
```

```shell
# 在主节点上启动DP Router
# 将4台服务器ip分别填入[IP_Node0],[IP_Node1],[IP_Node2],[IP_Node3]
# 主节点ip填入[master node ip]
python3 -m vllm_utils.router \
 --server-urls "http://[IP_Node0]:7555 \
  http://[IP_Node0]:7556 \
  http://[IP_Node1]:7555 \
  http://[IP_Node1]:7556 \
  http://[IP_Node2]:7555 \
  http://[IP_Node2]:7556 \
  http://[IP_Node3]:7555 \
  http://[IP_Node3]:7556" \
 --host [master node ip] \
 --port 8002 \
 --model [path of deepseek-v3-awq]
```

```shell
# 在主节点上启动客户端
# 主节点ip填入[master node ip]
python3 -m vllm_utils.benchmark_serving \
 --model [path of deepseek-v3-awq] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 64 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore-eos \
 --base-url http://[master node ip]:8002 \
 --extra-body '{"priority": 1}'
```
注：
*  本模型支持的`max-model-len`为163840，在32张S60上最大可支持65536；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；