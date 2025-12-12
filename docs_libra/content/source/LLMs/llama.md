## llama

### Meta-Llama-3-70B

#### 模型下载
*  url: [Meta-Llama-3-70B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-70B)

*  branch: `master`

*  commit id: `1130551d`

将上述 url 路径下的内容全部下载到 `Meta-Llama-3-70B` 文件夹中。

#### requirements

```shell
python3 -m pip install transformers==4.55.2
```

#### 环境变量
```
# v1 engine

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCHGCU_INDUCTOR_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_EFML_BASED_GCU_CHECK=1
```

#### 在线测试

##### 启动服务端

```shell
vllm serve [path of Meta-Llama-3-70B] \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --disable-log-requests \
        --block-size=64 \
        --dtype=bfloat16 \
        --tokenizer [path of Meta-Llama-3-70B] \
        --trust-remote-code \
        --gpu-memory-utilization=0.9 \
        --no-enable-prefix-caching \
        --async-scheduling
```

##### 启动客户端

```shell
curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-72tkvudyGLPMi" \
  -d '{"max_tokens": 50,
       "prompt": "User: What is Deep Learning?\nAssistant:",
       "model": "[path of Meta-Llama-3-70B]",
       "stream": false}'
```

#### 性能测试

##### 启动服务端

```shell
vllm serve [path of Meta-Llama-3-70B] \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --disable-log-requests \
        --block-size=64 \
        --dtype=bfloat16 \
        --tokenizer [path of Meta-Llama-3-70B] \
        --trust-remote-code \
        --gpu-memory-utilization=0.9 \
        --no-enable-prefix-caching \
        --async-scheduling
```

##### 启动客户端

```shell
vllm bench serve \
        --backend vllm  \
        --dataset-name random  \
        --model [path of Meta-Llama-3-70B] \
        --num-prompts 3 \
        --max-concurrency 1 \
        --random-input-len 2048 \
        --random-output-len 1024 \
        --trust-remote-code \
        --ignore_eos
```

### Meta-Llama-3-8B

#### 模型下载
*  url: [Meta-Llama-3-8B](https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8b)

*  branch: `master`

*  commit id: `3c106cb0`

将上述 url 路径下的内容全部下载到 `Meta-Llama-3-8B` 文件夹中。

#### requirements

```shell
python3 -m pip install transformers==4.55.2
```

#### 环境变量
```
# v1 engine

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCHGCU_INDUCTOR_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_EFML_BASED_GCU_CHECK=1
```

#### 在线测试

##### 启动服务端

```shell
vllm serve [path of Meta-Llama-3-8B] \
    --max-model-len 8192 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --async-scheduling
```

##### 启动客户端

```shell
curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 50,
       "prompt": "User: What is Deep Learning?\nAssistant:",
       "model": [path of Meta-Llama-3-8B],
       "stream": false}'
```

#### 性能测试

##### 启动服务端

```shell
vllm serve [path of Meta-Llama-3-8B] \
    --max-model-len 8192 \
    --disable-log-requests \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --dtype=bfloat16 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --async-scheduling
```

##### 启动客户端

```shell
vllm bench serve \
    --backend vllm  \
    --dataset-name random \
    --model [path of Meta-Llama-3-8B] \
    --num-prompt 3 \
    --max-concurrency 1 \
    --random-input-len 2048 \
    --random-output-len 1024 \
    --trust-remote-code \
    --ignore_eos
```