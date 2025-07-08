## Qwen

### Qwen2-72B-padded-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen2-72B-padded-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen2-72B-padded-w8a16_gptq`文件夹中。
- `Qwen2-72B-padded-w8a16_gptq`目录结构如下所示：

```shell
Qwen2-72B-padded-w8a16_gptq
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
├── tops_quantize_info.json
└── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Qwen2-72B-padded-w8a16_gptq] \
    --tokenizer=[path of Qwen2-72B-padded-w8a16_gptq] \
    --num-prompts 1 \
    --max-model-len=32768 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.9
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-72B-padded-w8a16_gptq]\
    --tensor-parallel-size 8 \
    --max-model-len=32768 \
    --input-len=8000 \
    --output-len=8000  \
    --dtype=float16 \
    --device gcu \
    --num-prompts=1  \
    --block-size=64 \
    --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;



### Qwen1.5-110B-Chat-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`QWen1.5-110B-Chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`QWen1.5-110B-Chat_w8a16_gptq`文件夹中。
- `QWen1.5-110B-Chat_w8a16_gptq`目录结构如下所示：

```shell
QWen1.5-110B-Chat_w8a16_gptq/
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── merges.txt
            ├── tops_quantize_info.json
            └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-110B-Chat] \
 --tensor-parallel-size=8 \
 --output-len=256 \
 --demo=te \
 --dtype=float16 \
 --block-size=64 \
 --num-prompts=1 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-110B-Chat] \
 --max-model-len=32768 \
 --tokenizer=[path of Qwen1.5-110B-Chat] \
 --input-len=1000 \
 --output-len=3000 \
 --tensor-parallel-size=8 \
 --num-prompts=1 \
 --block-size=16 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Qwen2-72B-w8a8c8

本模型推理及性能测试需要4张enflame gcu。

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen2-72B-w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen2-72B-w8a8c8`文件夹中。
- `Qwen2-72B-w8a8c8`目录结构如下所示：

```shell
Qwen2-72B-w8a8c8/

```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2-72B-w8a8c8] \
 --max-model-len=32768 \
 --demo=te \
 --dtype=bfloat16 \
 --tensor-parallel-size=4 \
 --quantization-param-path=[path of Qwen2-72B-w8a8c8] \
 --kv-cache-dtype=int8 \
 --output-len=256
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of Qwen2-72B-w8a8c8]  \
 --max-model-len=32768  \
 --tensor-parallel-size=4 \
 --disable-log-requests  \
 --gpu-memory-utilization=0.9  \
 --block-size=64 \
 --dtype=bfloat16 \
 --kv-cache-dtype=int8 \
 --quantization-param-path=[path of Qwen2-72B-w8a8c8]


# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend=vllm  \
 --dataset-name=random  \
 --model=[path of Qwen2-72B-w8a8c8]  \
 --num-prompts=1  \
 --random-input-len=3000 \
 --random-output-len=1000 \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Qwen2.5-32B-Instruct-GPTQ-Int8_w8a16

#### 模型下载
*  url: [Qwen2.5-32B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `996af7d8`

将上述url设定的路径下的内容全部下载到`Qwen2.5-32B-Instruct-GPTQ-Int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size 2 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-72B-Instruct-GPTQ-Int8_w8a16

#### 模型下载
*  url: [Qwen2.5-72B-Instruct-GPTQ-Int8](https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `ce39716d`

将上述url设定的路径下的内容全部下载到`Qwen2.5-72B-Instruct-GPTQ-Int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size 4 \
 --max-seq-len-to-capture 32768 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-0.5B-Instruct

#### 模型下载
*  url: [Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files)

*  branch: `master`

*  commit id: `42b20c45`

将上述url设定的路径下的内容全部下载到`Qwen2.5-0.5B-Instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-0.5B-Instruct] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-0.5B-Instruct] \
 --tensor-parallel-size 1 \
 --max-seq-len-to-capture=32768 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-0.5B-Instruct] \
 --num-prompts 1 \
 --random-input-len 4096 \
 --random-output-len 4096 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-Coder-32B-Instruct
#### 模型下载
*  url: [Qwen2.5-Coder-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-Coder-32B-Instruct/files)

*  branch: `master`

*  commit id: `19b16075`

将上述url设定的路径下的内容全部下载到`Qwen2.5-Coder-32B-Instruct`文件夹中。

#### requirements

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
```

#### 批量离线推理
```shell
 python3.10 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-coder-32b-instruct] \
 --demo=te \
 --tensor-parallel-size 8 \
 --max-model-len=32768 \
 --output-len=128 \
 --dtype=bfloat16 \
 --device gcu \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code
```

#### 性能测试
```shell
 python3.10 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of Qwen2.5-coder-32b-instruct] \
 --tensor-parallel-size 8 \
 --max-model-len=32768 \
 --input-len=9216 \
 --output-len=9216 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --backend vllm

```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### QwQ-32B
#### 模型下载
*  url: [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/files)

*  branch: `master`

*  commit id: `887ddbd72be5bed61cd702b87cd2fc25e65a708d`

将上述url设定的路径下的内容全部下载到`QwQ-32B`文件夹中。

#### 批量离线推理
```shell
 python3.10 -m vllm_utils.benchmark_test \
 --model=[path of QwQ-32B] \
 --demo=te \
 --tensor-parallel-size 4 \
 --max-model-len=32768 \
 --output-len=1024 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of QwQ-32B] \
  --num-scheduler-steps=16 \
  --tensor-parallel-size 4 \
  --max-seq-len-to-capture=32768 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16


# 启动客户端
  python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of QwQ-32B] \
  --num-prompts 1 \
  --random-input-len 512 \
  --random-output-len 512 \
  --trust-remote-code \
  --ignore_eos \
  --strict-in-out-len \
  --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### QWen3-30B-A3B

#### 模型下载
*  url: [QWen3-30B-A3B](https://www.modelscope.cn/models/Qwen/QWen3-30B-A3B/files)

*  branch: `master`

*  commit id: `e34b3e98`

将上述url设定的路径下的内容全部下载到`QWen3-30B-A3B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.51.3
```

#### 批量离线推理
```shell
  python3 -m vllm_utils.benchmark_test \
  --model [path of QWen3-30B-A3B] \
  --tensor-parallel-size=2 \
  --max-model-len=32768 \
  --output-len=128 \
  --demo=te \
  --dtype=bfloat16 \
  --device gcu
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of QWen3-30B-A3B] \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --disable-log-requests \
  --block-size=64 \
  --dtype=bfloat16 \
  --device gcu \
  --trust-remote-code \
  --gpu-memory-utilization=0.9 \
  --enable-chunked-prefill \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'


# 启动客户端
 python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of QWen3-30B-A3B] \
  --num-prompts 1 \
  --random-input-len 130048 \
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

### QWen3-30B-A3B-AWQ

#### 模型下载
*  url: [QWen3-30B-A3B-AWQ](https://modelscope.cn/models/swift/Qwen3-30B-A3B-AWQ/files)

*  branch: `master`

*  commit id: `3441b6ac`

将上述url设定的路径下的内容全部下载到`QWen3-30B-A3B-AWQ`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.51.3
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of QWen3-30B-A3B-AWQ] \
 --tensor-parallel-size 2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code \
 --quantization=moe_wna16_gcu
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of QWen3-30B-A3B-AWQ] \
 --tensor-parallel-size 2 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefill \
 --quantization=moe_wna16_gcu \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of QWen3-30B-A3B-AWQ] \
 --num-prompts 1 \
 --random-input-len 130048 \
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

### Qwen3-32B-AWQ

#### 模型下载
*  url: [Qwen3-32B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ)

*  branch: `master`

*  commit id: `f8fa721f`

将上述url设定的路径下的内容全部下载到`Qwen3-32B-AWQ`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.51.3
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of Qwen3-32B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len=40960 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen3-32B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefill \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-32B-AWQ] \
 --num-prompts 1 \
 --random-input-len 130048 \
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

### Qwen3-32B

#### 模型下载
*  url: [Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)

*  branch: `master`

*  commit id: `bc4962f6`

将上述url设定的路径下的内容全部下载到`Qwen3-32B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.52.3
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of Qwen3-32B] \
 --tensor-parallel-size 4 \
 --max-model-len=40960 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen3-32B] \
 --tensor-parallel-size 4 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefill


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-32B] \
 --num-prompts 32 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen3-235B-A22B-AWQ

#### 模型下载
*  url: [Qwen3-235B-A22B-AWQ](https://www.modelscope.cn/models/cognitivecomputations/Qwen3-235B-A22B-AWQ/)

*  branch: `master`

*  commit id: `56eac61f`

将上述url设定的路径下的内容全部下载到`Qwen3-235B-A22B-AWQ`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.52.3
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of Qwen3-235B-A22B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len=102400 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code \
 --enable-chunked-prefill \
 --block-size=64 \
 --quantization moe_wna16_gcu
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen3-235B-A22B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len 102400 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefill \
 --quantization=moe_wna16_gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-235B-A22B-AWQ] \
 --num-prompts 32 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;