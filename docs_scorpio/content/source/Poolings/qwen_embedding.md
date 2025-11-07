## Qwen

### Qwen3-Embedding-8b

#### 模型下载
*  url: [qwen3-Embedding-8b](https://huggingface.co/Qwen/Qwen3-Embedding-8B)

*  branch: `main`

*  commit id: `1d8ad4c`

将上述 url 路径下的内容全部下载到 `Qwen3-Embedding-8b` 文件夹中。

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
vllm serve "[path of Qwen3-Embedding-8b]" \
 --dtype=bfloat16 \
 --max-model-len 32768 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --trust-remote-code \
 --no-enable-prefix-caching \
 --async-scheduling

# 启动客户端
curl -X POST \
http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of Qwen3-Embedding-8b]",
        "input": [
            "text1",
            "text2"
        ]
      }'
```


#### 性能测试


```shell
# 启动服务端
vllm serve [path of Qwen3-Embedding-8b] \
 --dtype=bfloat16 \
 --max-model-len 32768 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --trust-remote-code \
 --async-scheduling \
 --no-enable-prefix-caching

# 启动客户端
python -m vllm_utils.benchmark_embedding_rerank \
 --test-type embedding \
 --api-url http://localhost:8000/v1/embeddings \
 --model [path of Qwen3-Embedding-8b] \
 --input-len 1024 \
 --total-requests 256 \
 --request-rate inf \
 --max-concurrency 1 \
 --tokenizer [path of Qwen3-Embedding-8b]
```

