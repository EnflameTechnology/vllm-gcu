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

