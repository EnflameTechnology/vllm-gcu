## bge

### bge-reranker-v2-m3

#### 模型下载
*  url: [bge-reranker-v2-m3](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3)

*  branch: `master`

*  commit id: `e099d4b9`

将上述 url 路径下的内容全部下载到 `bge-reranker-v2-m3` 文件夹中。
注：需要安装以下依赖：
```
python3 -m pip install transformers==4.57.1 beir==2.2.0
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
vllm serve [path of bge-reranker-v2-m3] \
    --served-model-name bge-reranker-v2-m3  \
    --task embed \
    --trust-remote-code \
    --port 6343 \
    --dtype=bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --trust-remote-code \
    --no-enable-prefix-caching


# 启动客户端

curl -X POST \
http://localhost:6343/rerank \
  -H "Content-Type: application/json" \
  -d '{
        "model": "bge-reranker-v2-m3",
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

```shell
# 启动服务端
vllm serve [path of bge-reranker-v2-m3] \
    --served-model-name bge-reranker-v2-m3  \
    --task embed \
    --trust-remote-code \
    --port 6343 \
    --dtype=bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --block-size=64 \
    --no-enable-prefix-caching \
    --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_embedding_rerank \
    --tokenizer [path of bge-reranker-v2-m3] \
    --trust-remote-code \
    --test-type rerank \
    --api-url http://localhost:6343/rerank \
    --model bge-reranker-v2-m3 \
    --input-len 1024 \
    --total-requests 256 \
    --query-len 20 \
    --num-docs 1 \
    --max-concurrency 1
```
