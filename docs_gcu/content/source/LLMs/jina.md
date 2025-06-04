## jina

### jina-reranker-v2-base-multilingual

#### 模型下载
*  url: [jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual/)

*  branch: `main`

*  commit id: `eed787badf7784e1a25c0eaa428627c8cbef511e`

将上述 url 路径下的内容全部下载到 `jina-reranker-v2-base-multilingual` 文件夹中。

#### requirement
```
python3 -m pip install transformers==4.52.4 beir==2.1.0
```

#### 离线推理

Server:

```shell
python3 -m vllm.entrypoints.openai.api_server  \
    --model [jina-reranker-v2-base-multilingual] \
    --served-model-name jina-reranker-v2-base-multilingual  \
    --task embed \
    --trust-remote-code \
    --port 6343
```

Client:

```shell
curl -X POST \
http://localhost:6343/rerank \
  -H "Content-Type: application/json" \
  -d '{
        "model": "jina-reranker-v2-base-multilingual",
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
python3 -m vllm.entrypoints.openai.api_server  \
    --model [jina-reranker-v2-base-multilingual] \
    --served-model-name jina-reranker-v2-base-multilingual  \
    --task embed \
    --trust-remote-code \
    --port 6343
```

Client:

```shell
python3 -m vllm_utils.benchmark_embedding_rerank \
    --tokenizer [jina-reranker-v2-base-multilingual] \
    --trust-remote-code \
    --test-type rerank \
    --api-url http://localhost:6343/rerank \
    --model jina-reranker-v2-base-multilingual \
    --input-len 100 \
    --total-requests 1000 \
    --query-len 20 \
    --num-docs 100
```