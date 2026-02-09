## Qwen

### Qwen3-32B

#### 模型下载
*  url: [Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)

*  branch: `master`

*  commit id: `bc4962f6`

将上述url设定的路径下的内容全部下载到`Qwen3-32B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
# 启动服务端
vllm serve "[path of Qwen3-32B]" \
        --tensor-parallel-size 1 \
        --max-model-len 40960 \
        --gpu-memory-utilization 0.9 \
        --block-size=64 \
        --dtype=bfloat16 \
        --served-model-name Qwen3-32B \
        --disable-log-requests \
        --no-enable-prefix-caching \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-batched-tokens 1024


# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                        {
                            "role": "user",
                            "content": "What is Deep Learning?"
                        }
                    ],
        "model":"Qwen3-32B",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-32B]" \
        --tensor-parallel-size 1 \
        --max-model-len 40960 \
        --gpu-memory-utilization 0.9 \
        --block-size=64 \
        --dtype=bfloat16 \
        --served-model-name Qwen3-32B \
        --disable-log-requests \
        --no-enable-prefix-caching \
        --trust-remote-code \
        --enable-chunked-prefill \
        --max-num-batched-tokens 1024 \
        --async-scheduling


# 启动客户端
vllm bench serve \
        --dataset-name random \
        --model [path of Qwen3-32B] \
        --num-prompt 10 \
        --max-concurrency 1 \
        --random-input-len 2048 \
        --random-output-len 1024 \
        --trust-remote-code \
        --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-32B-FP8

#### 模型下载
*  url: [Qwen3-32B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-32B-FP8)

*  branch: `master`

*  commit id: `8c192d01`

将上述url设定的路径下的内容全部下载到`Qwen3-32B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-32B-FP8]" \
    --block-size=64 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --max-model-len 40960 \
    --trust-remote-code

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
        "model":"[path of Qwen3-32B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-32B-FP8]" \
    --block-size=64 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --max-model-len 40960 \
    --trust-remote-code

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-32B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-14B

#### 模型下载
*  url: [Qwen3-14B](https://www.modelscope.cn/models/Qwen/Qwen3-14B)

*  branch: `master`

*  commit id: `6b837d17`

将上述url设定的路径下的内容全部下载到`Qwen3-14B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-14B]" \
    --block-size=64 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --max-model-len 40960 \
    --trust-remote-code

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
        "model":"[path of Qwen3-14B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-14B]" \
    --block-size=64 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 40960 \
    --trust-remote-code

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-14B] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-14B-FP8

#### 模型下载
*  url: [Qwen3-14B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-14B-FP8)

*  branch: `master`

*  commit id: `67a1c549`

将上述url设定的路径下的内容全部下载到`Qwen3-14B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-14B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-14B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-14B-FP8]" \
    --block-size=64 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 40960 \
    --trust-remote-code 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-14B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-8B

#### 模型下载
*  url: [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)

*  branch: `master`

*  commit id: `26028140`

将上述url设定的路径下的内容全部下载到`Qwen3-8B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-8B]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-8B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-8B]" \
    --block-size=64 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 40960 \
    --trust-remote-code 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-8B] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-8B-FP8

#### 模型下载
*  url: [Qwen3-8B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-8B-FP8)

*  branch: `master`

*  commit id: `82f329a8`

将上述url设定的路径下的内容全部下载到`Qwen3-8B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-8B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-8B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-8B-FP8]" \
    --block-size=64 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 40960 \
    --trust-remote-code 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-8B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-4B

#### 模型下载
*  url: [Qwen3-4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B)

*  branch: `master`

*  commit id: `2c54d5a0`

将上述url设定的路径下的内容全部下载到`Qwen3-4B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-4B]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-4B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-4B]" \
    --block-size=64 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --max-model-len 40960 \
    --trust-remote-code 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-4B] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-4B-FP8

#### 模型下载
*  url: [Qwen3-4B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-4B-FP8)

*  branch: `master`

*  commit id: `dcfda577`

将上述url设定的路径下的内容全部下载到`Qwen3-4B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-4B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-4B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-4B-FP8]" \
    --block-size=64 \
    --no-enable-prefix-caching \
    --async-scheduling \
    --max-model-len 40960 \
    --trust-remote-code 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-4B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-1.7B

#### 模型下载
*  url: [Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B)

*  branch: `master`

*  commit id: `4855588e`

将上述url设定的路径下的内容全部下载到`Qwen3-1.7B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
# 启动服务端
vllm serve "[path of Qwen3-1.7B]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-1.7B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-1.7B]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
    --no-enable-prefix-caching

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-1.7B] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-1.7B-FP8

#### 模型下载
*  url: [Qwen3-1.7B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B-FP8)

*  branch: `master`

*  commit id: `a615075c`

将上述url设定的路径下的内容全部下载到`Qwen3-1.7B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-1.7B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-1.7B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-1.7B-FP8]" \
    --trust-remote-code \
    --block-size=64 \
    --async-scheduling \
    --max-model-len 40960 \
    --no-enable-prefix-caching
 

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-1.7B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-0.6B

#### 模型下载
*  url: [Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)

*  branch: `master`

*  commit id: `09b42cad`

将上述url设定的路径下的内容全部下载到`Qwen3-0.6B`文件夹中。

注：需要安装以下依赖：
#### requirements
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
# 启动服务端
vllm serve "[path of Qwen3-0.6B]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-0.6B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-0.6B]" \
    --trust-remote-code \
    --block-size=64 \
    --async-scheduling \
    --max-model-len 40960 \
    --no-enable-prefix-caching

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-0.6B] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen3-0.6B-FP8

#### 模型下载
*  url: [Qwen3-0.6B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-FP8)

*  branch: `master`

*  commit id: `235c7885`

将上述url设定的路径下的内容全部下载到`Qwen3-0.6B-FP8`文件夹中。

注：需要安装以下依赖：
#### requirements
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
vllm serve "[path of Qwen3-0.6B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
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
        "model":"[path of Qwen3-0.6B-FP8]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen3-0.6B-FP8]" \
    --block-size=64 \
    --trust-remote-code \
    --async-scheduling \
    --max-model-len 40960 \
    --no-enable-prefix-caching

# 启动客户端
vllm bench serve \
    --dataset-name random \
    --model [path of Qwen3-0.6B-FP8] \
    --num-prompts 40 \
    --max-concurrency 4 \
    --random-input-len 128 \
    --random-output-len 4096 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为40960；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### Qwen2-72B-Instruct
#### 模型下载
*  url: [Qwen2-72B-Instruct](https://modelscope.cn/models/Qwen/Qwen2-72B-Instruct)

*  branch: `master`

*  commit id: `66eb39b04e22cffbe2377447d5a7b13c0b4dd814`

将上述url设定的路径下的内容全部下载到`Qwen2-72B-Instruct`文件夹中。
注：需要安装以下依赖：
#### requirements
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
 # 启动服务端
vllm serve "[path of Qwen2-72B-Instruct]"  \
        --tokenizer="[path of Qwen2-72B-Instruct]"  \
        --dtype=bfloat16 \
        --max-model-len=32768 \
        --tensor-parallel-size=2 \
        --block-size=64 \
        --disable-log-requests \
        --gpu-memory-utilization=0.9 \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --served-model-name Qwen2-72B-Instruct \
        --enable-chunked-prefill

# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                       {
                           "role": "system",
                           "content": "You are a helpful assistant."
                       }
                    ],
        "model":"Qwen2-72B-Instruct",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Qwen2-72B-Instruct]"  \
        --dtype=bfloat16 \
        --max-model-len=32768 \
        --tensor-parallel-size=2 \
        --block-size=64 \
        --disable-log-requests \
        --gpu-memory-utilization=0.9 \
        --no-enable-prefix-caching \
        --async-scheduling \
        --served-model-name Qwen2-72B-Instruct


# 启动客户端
vllm bench serve \
        --dataset-name random \
        --model Qwen2-72B-Instruct \
        --tokenizer [path of Qwen2-72B-Instruct] \
        --num-prompts 10 \
        --max-concurrency 1 \
        --random-input-len 2048 \
        --random-output-len 1024 \
        --trust-remote-code \
        --ignore_eos \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --metric-percentiles 25,50,75,90,95,99,100 \
        --save-result
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen3-30B-A3B-Instruct-2507-FP8
#### 模型下载
*  url: [Qwen3-30B-A3B-Instruct-2507-FP8](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)

*  branch: `master`

*  commit id: `e6c8b0cf`

将上述url设定的路径下的内容全部下载到`Qwen3-30B-A3B-Instruct-2507-FP8`文件夹中。

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
vllm serve "[path of Qwen3-30B-A3B-Instruct-2507-FP8]" \
 --tensor-parallel-size 1 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --dtype=bfloat16 \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code \
 --served-model-name Qwen3-30B-A3B-Instruct-2507-FP8 \
 --max-model-len 262144

# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen3-30B-A3B-Instruct-2507-FP8",
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
vllm serve "[path of Qwen3-30B-A3B-Instruct-2507-FP8]" \
 --tensor-parallel-size 1  \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --dtype=bfloat16  \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code \
 --served-model-name Qwen3-30B-A3B-Instruct-2507-FP8 \
 --max-model-len 262144 \
 --quantization=fp8 \
 --cuda-graph-sizes 1 2 3 4 5 6 7 8 12 13 16 20 24 25 28 32 36 40 44 48 52 56 60 64


# 启动客户端
vllm bench serve \
 --model Qwen3-30B-A3B-Instruct-2507-FP8 \
 --dataset-name random \
 --random-input-len 128 \
 --random-output-len 128 \
 --num-prompts 10 \
 --max-concurrency 1 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl,e2el \
 --metric-percentiles 25,50,75,90,95,99,100
```
注：
*  本模型支持的`max-model-len`为262144；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

