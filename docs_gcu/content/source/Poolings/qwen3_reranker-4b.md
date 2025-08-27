## qwen

### qwen3-reranker-4b

#### 模型下载
*  url: [qwen3-reranker-4b](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

*  branch: `main`

*  commit id: ``

将上述 url 路径下的内容全部下载到 `qwen3-reranker-4b` 文件夹中。

### 环境变量
```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### requirement
```
python3 -m pip install transformers==4.51.3 beir==2.2.0
```

#### 离线推理

Server:

```shell
vllm serve [path of qwen3-reranker-4b] \
    --served-model-name qwen3-reranker-4b  \
    --task embed \
    --trust-remote-code \
    --port 6343 \
    --dtype=bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --trust-remote-code
```

Client:

```shell
curl -X POST \
http://localhost:6343/rerank \
  -H "Content-Type: application/json" \
  -d '{
        "model": "qwen3-reranker-4b",
        "query": "人工智能在医疗领域的应用现状",
        "documents": [
            "AI医学影像识别可辅助医生诊断肺癌、乳腺癌等疾病，准确率超95%",
            "自然语言处理技术用于电子病历分析，提升病历检索效率300%",
            "深度学习是人工智能的一个分支，基于神经网络"
        ],
        "normalize": false
      }'
```


#### 性能测试

Server:

```shell
vllm serve [path of qwen3-reranker-4b] \
    --served-model-name qwen3-reranker-4b  \
    --task embed \
    --trust-remote-code \
    --port 6343 \
    --dtype=bfloat16 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --trust-remote-code
```

Client:

```shell
python3 -m vllm_utils.benchmark_embedding_rerank \
    --tokenizer [path of qwen3-reranker-4b] \
    --trust-remote-code \
    --test-type rerank \
    --api-url http://localhost:6343/rerank \
    --model qwen3-reranker-4b \
    --input-len 100 \
    --total-requests 1000 \
    --query-len 20 \
    --num-docs 100
```
