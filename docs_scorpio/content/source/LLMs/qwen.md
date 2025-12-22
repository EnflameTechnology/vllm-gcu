## Qwen

### Qwen3-32B

#### 模型下载
*  url: [Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)

*  branch: `master`

*  commit id: `bc4962f6`

将上述url设定的路径下的内容全部下载到`Qwen3-32B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

```

#### 在线测试
```shell
# 启动服务端
vllm serve "[path of Qwen3-32B]" \
 --tensor-parallel-size 4 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --async-scheduling \
 --no-enable-prefix-caching

# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                        {
                            "role": "system",
                             "content": "You are a helpful assistant."
                         },
                         {
                             "role":"user",
                             "content":"李白是谁？"
                         }
                     ],
        "model":"[path of Qwen3-32B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-32B]" \
 --tensor-parallel-size 4 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --async-scheduling \
 --no-enable-prefix-caching


# 启动客户端
vllm bench serve \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-32B] \
 --num-prompts 32 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore_eos 
```
注：
*  本模型支持的`max-model-len`为131072；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### QwQ-32B
#### 模型下载
*  url: [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/files)

*  branch: `master`

*  commit id: `887ddbd72be5bed61cd702b87cd2fc25e65a708d`

将上述url设定的路径下的内容全部下载到`QwQ-32B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
 # 启动服务端
  vllm serve "[path of QwQ-32B]" \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --async-scheduling \
  --no-enable-prefix-caching

# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                       {
                           "role": "system",
                           "content": "You are a helpful assistant."
                       },
                       {
                           "role":"user",
                           "content":"李白是谁？"
                       }
                   ],
        "model":"[path of QwQ-32B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve "[path of QwQ-32B]" \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --async-scheduling \
  --no-enable-prefix-caching


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of QwQ-32B] \
  --num-prompts 1 \
  --random-input-len 512 \
  --random-output-len 512 \
  --trust-remote-code \
  --ignore_eos 
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen3-Next-80B-A3B-Instruct

#### 模型下载
*  url: [Qwen3-Next-80B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct/)

*  branch: `master`

*  commit id: `34fec46d`

将上述url设定的路径下的内容全部下载到`Qwen3-Next-80B-A3B-Instruct`文件夹中。

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of Qwen3-Next-80B-A3B-Instruct]" \
	--tensor-parallel-size 8 \
	--dtype=bfloat16 \
	--trust-remote-code \
	--block-size=256 \
	--max-model-len=262144 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--compilation_config '{"cudagraph_mode":"FULL"}' \
	--gpu-memory-utilization 0.8

# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of Qwen3-Next-80B-A3B-Instruct]",
        "prompt": [
          "请介绍北京的旅游景点",
          "介绍一下大熊猫",
          "晚上睡不着应该怎么办",
          "李白的代表作有哪些？"
        ],
        "max_tokens": 128,
        "temperature": 0
     }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-Next-80B-A3B-Instruct]" \
	--tensor-parallel-size 8 \
	--dtype=bfloat16 \
	--trust-remote-code \
	--block-size=256 \
	--max-model-len=262144 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--compilation_config '{"cudagraph_mode":"FULL"}' \
	--gpu-memory-utilization 0.8


# 启动客户端
vllm bench serve --model "[path of Qwen3-Next-80B-A3B-Instruct]" \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 4 \
 --max-concurrency 1 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl \
 --metric-percentiles 25,50,75,90,99,100
```
注：
*  本模型支持的`max-model-len`为262144；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen3-30B-A3B

#### 模型下载
*  url: [Qwen3-30B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)

*  branch: `master`

*  commit id: `e34b3e98`

将上述url设定的路径下的内容全部下载到`Qwen3-30B-A3B`文件夹中。

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of Qwen3-30B-A3B]" \
 --max-model-len 32768 \
 --tensor-parallel-size 2 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code

# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of Qwen3-30B-A3B]",
        "prompt": [
          "请介绍北京的旅游景点",
          "介绍一下大熊猫",
          "晚上睡不着应该怎么办",
          "李白的代表作有哪些？"
        ],
        "max_tokens": 128,
        "temperature": 0
     }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-30B-A3B]" \
 --max-model-len 32768 \
 --tensor-parallel-size 2 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code


# 启动客户端
vllm bench serve --model "[path of Qwen3-30B-A3B]" \
 --dataset-name random \
 --random-input-len 1000 \
 --random-output-len 700 \
 --num-prompts 64 \
 --max-concurrency 32 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl \
 --metric-percentiles 25,50,75,90,99,100
```
注：
*  本模型支持的`max-model-len`为40960；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen3-32B-AWQ

#### 模型下载
*  url: [Qwen3-32B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ/)

*  branch: `master`

*  commit id: `196ed22e`

将上述url设定的路径下的内容全部下载到`Qwen3-32B-AWQ`文件夹中。

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of Qwen3-32B-AWQ]" \
 --max-model-len 131072 \
 --tensor-parallel-size 4 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code

# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of Qwen3-32B-AWQ]",
        "prompt": [
          "请介绍北京的旅游景点",
          "介绍一下大熊猫",
          "晚上睡不着应该怎么办",
          "李白的代表作有哪些？"
        ],
        "max_tokens": 128,
        "temperature": 0
     }'

```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-32B-AWQ]" \
 --max-model-len 131072 \
 --tensor-parallel-size 4 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code


# 启动客户端
vllm bench serve --model "[path of Qwen3-30B-A3B]" \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 64 \
 --max-concurrency 32 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl   \
 --metric-percentiles 25,50,75,90,99,100
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
`
### Qwen2-1.5B
#### 模型下载
*  url: [Qwen2-1.5B](https://modelscope.cn/models/Qwen/Qwen2-1.5B/files)

*  branch: `master`

*  commit id: `2f0ed2d6049f639abf50250b719a8a432ef0f283`

将上述url设定的路径下的内容全部下载到`Qwen2-1.5B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```

#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务端
  vllm serve [path of Qwen2-1.5B] \
    --served-model-name Qwen2-1.5B \
    --tensor-parallel-size 1\
    --max-model-len=32768 \
    --dtype=bfloat16 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --no-enable-prefix-caching


# 启动客户端
  curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 64,"prompt":"李白是谁","model":"Qwen2-1.5B","stop": null,"stream": false}'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen2-1.5B] \
    --tensor-parallel-size 1\
    --max-model-len=32768 \
    --disable-log-requests \
    --dtype=bfloat16 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --no-enable-prefix-caching \
    --async-scheduling


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen2-1.5B] \
  --num-prompts 16 \
  --max-concurrency 4 \
  --random-input-len 1024 \
  --random-output-len 32 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；


### Qwen2.5-32B

#### 模型下载
*  url: [Qwen2.5-32B](https://www.modelscope.cn/models/qwen/Qwen2.5-32B)

*  branch: `master`

*  commit id: `357d2bb7`

将上述url设定的路径下的内容全部下载到`Qwen2.5-32B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1

```
#### 环境变量

```
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of Qwen2.5-32B]" \
 --tokenizer=[path of Qwen2.5-32B] \
 --dtype=bfloat16 \
 --max-model-len=131072 \
 --served-model-name Qwen2.5-32B \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --disable-log-requests \
 --gpu-memory-utilization=0.9 \
 --trust-remote-code \
 --no-enable-prefix-caching

# 启动客户端
curl "http://127.0.0.1:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
    "max_tokens": 500,
    "prompt":["请介绍北京的旅游景点"],
    "model":"Qwen2.5-32B",
    "stop": null,
    "stream": false
    }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen2.5-32B]" \
 --tokenizer=[path of Qwen2.5-32B] \
 --dtype=bfloat16 \
 --max-model-len=131072 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
  --no-enable-prefix-caching \
 --trust-remote-code \
 --async-scheduling


# 启动客户端
vllm bench serve \
 --model [path of Qwen2.5-32B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 32 \
 --disable-log-requests \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --request-rate 1 \
 --trust-remote-code \
 --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen2.5-32b-Instruct-GPTQ-Int8

#### 模型下载
*  url: [Qwen2.5-32b-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8/)

*  branch: `master`

*  commit id: `fca9cc95`

将上述url设定的路径下的内容全部下载到`Qwen2.5-32b-Instruct-GPTQ-Int8`文件夹中。

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务器

vllm serve "[path of Qwen2.5-32b-Instruct-GPTQ-Int8]" \
 --tokenizer [path of Qwen2.5-32b-Instruct-GPTQ-Int8] \
 --max-model-len 32768 \
 --tensor-parallel-size 2 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --gpu-memory-utilization=0.9 \
 --trust-remote-code \
 --quantization moe_wna16_gcu

# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
         "max_tokens": 500,
         "messages": [
             {
                 "role": "system",
                 "content": "You are a helpful assistant."
             },
             {
                 "role":"user",
                 "content":"李白是谁？"
             }],
         "model":"[path of Qwen2.5-32b-Instruct-GPTQ-Int8]",
         "stop": null,
         "stream": false
     }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen2.5-32b-Instruct-GPTQ-Int8]" \
 --tokenizer [path of Qwen2.5-32b-Instruct-GPTQ-Int8] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=2 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --trust_remote_code \
 --quantization moe_wna16_gcu \
 --async-scheduling


# 启动客户端
vllm bench serve \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-32b-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen2-7B
#### 模型下载
*  url: [Qwen2-7B](https://modelscope.cn/models/Qwen/Qwen2-7B/files)

*  branch: `master`

*  commit id: `96127251daf38ea8aa588eac3d34faf64682c0e5`

将上述url设定的路径下的内容全部下载到`Qwen2-7B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```

#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 离线推理

```shell
# 启动服务端
  vllm serve [path of Qwen2-7B] \
    --served-model-name Qwen2-7B \
    --max-model-len=32768 \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching


# 启动客户端
  curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 64,"prompt":"李白是谁","model":"Qwen2-7B","stop": null,"stream": false}'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen2-7B] \
    --max-model-len=32768 \
    --disable-log-requests \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching \
    --async-scheduling


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen2-7B] \
  --max-concurrency 4 \
  --num-prompts 40 \
  --random-input-len 1024 \
  --random-output-len 32 \
  --request-rate 1 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`、`max-concurrency`和`num-prompts`可按需调整；

### Qwen3-30B-A3B-AWQ
#### 模型下载
*  url: [Qwen3-30B-A3B-AWQ](https://modelscope.cn/models/swift/Qwen3-30B-A3B-AWQ/files)

*  branch: `master`

*  commit id: `3441b6ac9596e224f77a319be3b4c6149029d0b3`

将上述url设定的路径下的内容全部下载到`Qwen3-30B-A3B-AWQ`文件夹中。

#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线推理

```shell
# 启动服务端
  vllm serve [path of Qwen3-30B-A3B-AWQ] \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --quantization=moe_wna16_gcu \
    --trust-remote-code \
    --port 8989


# 启动客户端
  curl "http://127.0.0.1:8989/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                        {
                            "role": "system",
                             "content": "You are a helpful assistant."
                         },
                         {
                             "role":"user",
                             "content":"李白是谁？"
                         }
                     ],
        "model":"[path of Qwen3-30B-A3B-AWQ]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen3-30B-A3B-AWQ] \
    --tensor-parallel-size 1 \
    --max-model-len 131072 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --quantization=moe_wna16_gcu \
    --trust-remote-code \
    --async-scheduling \
    --compilation_config '{"cudagraph_mode":"FULL"}' \
    --no-enable-prefix-caching \
    --port 8989 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --percentile-metrics ttft,tpot,itl   \
    --metric-percentiles 25,50,75,90,99,100


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen3-30B-A3B-AWQ] \
  --num-prompt 40 \
  --max-concurrency 4 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos \
  --port 8989
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`、`max-concurrency`和`num-prompts`可按需调整；

### Qwen3-235B-A22B-AWQ
#### 模型下载
*  url: [Qwen3-235B-A22B-AWQ](https://huggingface.co/QuixiAI/Qwen3-235B-A22B-AWQ/tree/main)

*  branch: `master`

*  commit id: `1df91c166baa937f2d571a9cece7a1037c1cc772`

将上述url设定的路径下的内容全部下载到`Qwen3-235B-A22B-AWQ`文件夹中。


#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCH_COMPILE_DISABLE=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线推理

```shell
# 启动服务端
  vllm serve [path of Qwen3-235B-A22B-AWQ] \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --quantization=moe_wna16_gcu \
    --trust-remote-code \
    --port 8989


# 启动客户端
  curl "http://127.0.0.1:8989/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                        {
                            "role": "system",
                             "content": "You are a helpful assistant."
                         },
                         {
                             "role":"user",
                             "content":"李白是谁？"
                         }
                     ],
        "model":"[path of Qwen3-235B-A22B-AWQ]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen3-235B-A22B-AWQ] \
    --tensor-parallel-size 4 \
    --max-model-len 131072 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --quantization=moe_wna16_gcu \
    --trust-remote-code \
    --async-scheduling \
    --compilation_config '{"cudagraph_mode":"FULL"}' \
    --no-enable-prefix-caching \
    --port 8989 \
    --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
    --percentile-metrics ttft,tpot,itl   \
    --metric-percentiles 25,50,75,90,99,100


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen3-235B-A22B-AWQ] \
  --num-prompt 40 \
  --max-concurrency 4 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos \
  --port 8989
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`、`max-concurrency`和`num-prompts`可按需调整；


### Qwen2-72B-Instruct-GPTQ-Int8
#### 模型下载
*  url: [Qwen2-72B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2-72B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `f7d561d4`

将上述url设定的路径下的内容全部下载到`Qwen2-72B-Instruct-GPTQ-Int8`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.52.3
```

#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 离线推理

```shell
# 启动服务端
  vllm serve [path of Qwen2-72B-Instruct-GPTQ-Int8] \
    --served-model-name Qwen2-72B-Instruct-GPTQ-Int8 \
    --max-model-len=32768 \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching


# 启动客户端
  curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 64,"prompt":"李白是谁","model":"Qwen2-72B-Instruct-GPTQ-Int8","stop": null,"stream": false}'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen2-72B-Instruct-GPTQ-Int8] \
    --max-model-len=32768 \
    --disable-log-requests \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching \
    --async-scheduling


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen2-72B-Instruct-GPTQ-Int8] \
  --max-concurrency 4 \
  --num-prompts 40 \
  --random-input-len 1024 \
  --random-output-len 32 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`、`max-concurrency`和`num-prompts`可按需调整；


### Qwen2.5-14B-Instruct-GPTQ-Int8
#### 模型下载
*  url: [Qwen2.5-14B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `a432359f08ca0e491450723a24de23b88c19ad5e`

将上述url设定的路径下的内容全部下载到`Qwen2.5-14B-Instruct-GPTQ-Int8`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```

#### 环境变量

```
# v1 engine
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 离线推理

```shell
# 启动服务端
  vllm serve [path of Qwen2.5-14B-Instruct-GPTQ-Int8] \
    --served-model-name Qwen2.5-14B-Instruct-GPTQ-Int8 \
    --max-model-len=32768 \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --trust-remote-code \
    --quantization gptq


# 启动客户端
  curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 64,"prompt":"李白是谁","model":"Qwen2.5-14B-Instruct-GPTQ-Int8","stop": null,"stream": false}'
```

#### 性能测试

```shell
# 启动服务端
  vllm serve [path of Qwen2.5-14B-Instruct-GPTQ-Int8] \
    --max-model-len=32768 \
    --block-size=64 \
    --dtype=bfloat16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9 \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --trust-remote-code \
    --quantization gptq


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen2.5-14B-Instruct-GPTQ-Int8] \
  --num-prompts 10 \
  --max-concurrency 1 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`、`max-concurrency`和`num-prompts`可按需调整；
