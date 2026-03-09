## Hunyuan-v1
### Hunyuan-0.5B-Instruct

#### 模型下载
*  url: [Hunyuan-0.5B-Instruct](https://huggingface.co/tencent/Hunyuan-0.5B-Instruct)

*  branch: `main`

*  commit id: `2359fb2`

将上述url设定的路径下的内容全部下载到`Hunyuan-0.5B-Instruct`文件夹中。
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
vllm serve "[path of Hunyuan-0.5B-Instruct]" \
 --tokenizer=[path of Hunyuan-0.5B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=262144 \
 --served-model-name Hunyuan-0.5B \
 --tensor-parallel-size=1 \
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
    "model":"Hunyuan-0.5B",
    "stop": null,
    "stream": false
    }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Hunyuan-0.5B-Instruct]" \
 --tokenizer=[path of Hunyuan-0.5B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=262144 \
 --tensor-parallel-size=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --trust-remote-code


# 启动客户端
vllm bench serve \
 --model [path of Hunyuan-0.5B-Instruct] \
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
* 本模型支持的`max-model-len`为262144；
* `input-len`、`output-len`和`num-prompts`可按需调整；

### Hunyuan-7B-Instruct

#### 模型下载
*  url: [Hunyuan-7B-Instruct](https://huggingface.co/tencent/Hunyuan-7B-Instruct)

*  branch: `main`

*  commit id: `6fd6ecb`

将上述url设定的路径下的内容全部下载到`Hunyuan-7B-Instruct`文件夹中。
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
export TOPS_STREAM_SCHEDULE_CREDIT=4
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of Hunyuan-7B-Instruct]" \
 --tokenizer=[path of Hunyuan-7B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --served-model-name Hunyuan-7B \
 --tensor-parallel-size=1 \
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
    "model":"Hunyuan-7B",
    "stop": null,
    "stream": false
    }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Hunyuan-7B-Instruct]" \
 --tokenizer=[path of Hunyuan-7B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --trust-remote-code


# 启动客户端
vllm bench serve \
 --model [path of Hunyuan-7B-Instruct] \
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
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；


### Hunyuan-7B-Instruct-W4A16-AWQ-C8

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Hunyuan-7B-Instruct-W4A16-AWQ-C8.tar`文件并解压，将压缩包内的内容全部拷贝到`Hunyuan-7B-Instruct-W4A16-AWQ-C8`文件夹中。
- `Hunyuan-7B-Instruct-W4A16-AWQ-C8`目录结构如下所示：

```shell
Hunyuan-7B-Instruct-W4A16-AWQ-C8
├── chat_template.jinja
├── config.json
├── generation_config.json
├── int8_kv_cache.safetensors
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── tops_quantize_info.json
```

注：需要安装以下依赖：
#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

#### 在线测试

```shell
# 启动服务器
vllm serve "[path of Hunyuan-7B-Instruct-W4A16-AWQ-C8]" \
  --port 8192 \
  --max-model-len 32768 \
  --block-size=64 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --kv_cache-dtype=int8 \
  --trust-remote-code  \
  --quantization awq

# 启动客户端
curl http://0.0.0.0:8192/v1/chat/completions -H 'Content-Type: application/json' \
-d '{
    "model": "[path of Hunyuan-7B-Instruct-W4A16-AWQ-C8]",
    "messages": [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "请按面积大小对四大洋进行排序，并给出面积最小的洋是哪一个？直接输出结果。"}]
        }
    ],
    "max_tokens": 2048,
    "temperature":0.7,
    "top_p": 0.6,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "stop_token_ids": [127960]
    }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of Hunyuan-7B-Instruct-W4A16-AWQ-C8]" \
  --port 8192 \
  --block-size=64 \
  --max-model-len 32768 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --kv_cache-dtype=int8 \
  --trust-remote-code 

# 启动客户端
vllm bench serve --model "[path of Hunyuan-7B-Instruct-W4A16-AWQ-C8]" \
  --port 8192   \
  --dataset-name random \
  --num-prompts 40 \
  --max-concurrency 4 \
  --random-input-len 2048 \
  --random-output-len 2048 \
  --trust-remote-code \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 25,50,75,90,99,100
```

注：
*  Hunyuan-7B-Instruct-W4A16-AWQ-C8模型支持的`max-model-len`为32k；
*  `input-len`、`output-len`和`num-prompts`可按需调整;