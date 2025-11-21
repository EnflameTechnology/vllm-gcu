## Qwen

### QwQ-32B
#### 模型下载
*  url: [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/files)

*  branch: `master`

*  commit id: `887ddbd72be5bed61cd702b87cd2fc25e65a708d`

将上述url设定的路径下的内容全部下载到`QwQ-32B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
python3 -m pip install triton==3.1.0
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
vllm serve [path of QwQ-32B] \
  --tensor-parallel-size 4 \
  --served-model-name QwQ-32B \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --no-enable-prefix-caching


# 启动客户端
  curl "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 64,"prompt":"李白是谁","model":"QwQ-32B","stop": null,"stream": false}'
```

#### serving模式

```shell
# 启动服务端
vllm serve [path of QwQ-32B] \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --no-enable-prefix-caching


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of QwQ-32B] \
  --num-prompts 16 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --request-rate 1 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen2-1.5B
#### 模型下载
*  url: [Qwen2-1.5B](https://modelscope.cn/models/Qwen/Qwen2-1.5B/files)

*  branch: `master`

*  commit id: `2f0ed2d6049f639abf50250b719a8a432ef0f283`

将上述url设定的路径下的内容全部下载到`Qwen2-1.5B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
python3 -m pip install triton==3.1.0
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
  vllm serve [path of Qwen2-1.5B] \
    --served-model-name Qwen2-1.5B \
    --tensor-parallel-size 1\
    --max-model-len=32768 \
    --dtype=bfloat16 \
    --device gcu \
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
    --device gcu \
    --block-size=64 \
    --no-enable-prefix-caching


# 启动客户端
  vllm bench serve \
  --backend vllm \
  --dataset-name random \
  --model [path of Qwen2-1.5B] \
  --num-prompts 16 \
  --random-input-len 1024 \
  --random-output-len 32 \
  --request-rate 1 \
  --trust-remote-code \
  --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
