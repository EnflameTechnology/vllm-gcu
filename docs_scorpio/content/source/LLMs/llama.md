## llama
### Meta-Llama-3-8B-AWQ

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3-8B-AWQ](https://huggingface.co/solidrust/Meta-Llama-3-8B-AWQ/tree/main)

*  branch: `main`

*  commit id: `bac14f8`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3-8B-AWQ`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
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
# 服务端
vllm serve "[path of Meta-Llama-3-8B-AWQ]" \
  --served-model-name Meta-Llama-3-8B-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 2048 \
  --max-model-len 8192 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation_config '{"cudagraph_mode":"FULL"}' \
  --trust-remote-code \
  --port 8990 \
  --chat-template "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'system' %}{{ '### System:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\\n'  + message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{{ '### Assistant:\\n' }}"


# 客户端
curl -X POST http://localhost:8990/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "Meta-Llama-3-8B-AWQ",
        "messages": [
          {
            "role": "system",
            "content": "You are a helpful, thorough, and talkative assistant. Always elaborate your answers with rich details and examples."
          },
          {
            "role": "user",
            "content": "Explain the concept of neural networks in detail."
          }
        ]
      }'
```

#### 性能测试

```shell
# 服务端
vllm serve "[path of Meta-Llama-3-8B-AWQ]" \
  --quantization awq \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation_config '{"cudagraph_mode":"FULL"}' \
  --trust-remote-code \
  --port 8990

 # 客户端
vllm bench serve --model [path of Meta-Llama-3-8B-Instruct] \
    --backend vllm \
    --base-url "http://127.0.0.1:8990" \
    --dataset-name random \
    --num-prompts 4 \
    --max-concurrency 40 \
    --random-input-len 2048 \
    --random-output-len 1024 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Meta-Llama-3-70B-AWQ

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3-70B-AWQ](https://modelscope.cn/models/TechxGenus-MS/Meta-Llama-3-70B-AWQ/files)

*  branch: `master`

*  commit id: `050b1b2e`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3-70B-AWQ`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
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
# 服务端
vllm serve "[path of Meta-Llama-3-70B-AWQ]" \
  --served-model-name Meta-Llama-3-70B-AWQ \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 2048 \
  --max-model-len 8192 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation_config '{"cudagraph_mode":"FULL"}' \
  --trust-remote-code \
  --port 8990


# 客户端
curl -X POST http://localhost:8990/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "Meta-Llama-3-70B-AWQ",
        "messages": [
          {
            "role": "system",
            "content": "You are a helpful, thorough, and talkative assistant. Always elaborate your answers with rich details and examples."
          },
          {
            "role": "user",
            "content": "Explain the concept of neural networks in detail."
          }
        ]
      }'
```

#### 性能测试

```shell
# 服务端
vllm serve "[path of Meta-Llama-3-70B-AWQ]" \
  --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --compilation_config '{"cudagraph_mode":"FULL"}' \
  --trust-remote-code \
  --port 8990

 # 客户端
vllm bench serve --model [path of Meta-Llama-3-70B-Instruct] \
    --backend vllm \
    --base-url "http://127.0.0.1:8990" \
    --dataset-name random \
    --num-prompts 4 \
    --max-concurrency 40 \
    --random-input-len 2048 \
    --random-output-len 1024 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Meta-Llama-3.1-8B-Instruct

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main/)

*  branch: `main`

*  commit id: `8c22764`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-8B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
python3 -m pip install triton==3.1.0
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
# 服务端
vllm serve "[path of Meta-Llama-3.1-8B-Instruct]" \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --block-size=64 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --no-enable-prefix-caching \
 --trust_remote_code 

# 客户端
curl "http://127.0.0.1:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{"max_tokens": 500,"messages": [{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content":"李白是谁？"}],"model":"/ard-data/pretrained_models/Meta-Llama-3.1-8B-Instruct/","stop": null,"stream": false}'
```

#### 性能测试

```shell

# 服务端
vllm serve "[path of Meta-Llama-3.1-8B-Instruct]" \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --no-enable-prefix-caching

 # 客户端
vllm bench serve --model [path of Meta-Llama-3.1-8B-Instruct] \
  --backend vllm \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len 8192 \
  --random-output-len 512 \
  --trust-remote-code \
  --ignore_eos 
```
注：
*  本模型支持的`max-model-len`为131072, 单张卡可跑32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
