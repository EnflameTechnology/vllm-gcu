## Qwen2.5
### Qwen2.5-32B

#### 模型下载
*  url: [Qwen2.5-32B](https://www.modelscope.cn/models/qwen/Qwen2.5-32B)

*  branch: `master`

*  commit id: `357d2bb7`

将上述url设定的路径下的内容全部下载到`Qwen2.5-32B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
python3 -m pip install triton==3.1.0

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

#### serving模式
```shell
# 启动服务器
vllm serve "[path of Qwen2.5-32B]" \
 --tokenizer=[path of Qwen2.5-32B] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --served-model-name Qwen2.5-32B \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --device=gcu \
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
    "model":"[path of Qwen2.5-32B]",
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
 --max-model-len=32768 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --device=gcu \
 --gpu-memory-utilization=0.9 \
  --no-enable-prefix-caching \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of Qwen2.5-32B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 32 \
 --disable-log-requests \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --request-rate 1 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
