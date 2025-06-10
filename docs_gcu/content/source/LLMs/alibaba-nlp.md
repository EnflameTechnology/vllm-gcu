## Alibaba-NLP

### gte-Qwen2-7B-instruct

#### 模型下载
*  url: [gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)

*  branch: `main`

*  commit id: `a8d08b3`

将上述 url 路径下的内容全部下载到 `gte-Qwen2-7B-instruct` 文件夹中。

#### 在线推理

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of gte-Qwen2-7B-instruct] \
 --dtype=float16 \
 --max-model-len 32768

# 启动客户端
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"[path of gte-Qwen2-7B-instruct]","input":["text1","text2"]}'
```


#### 性能测试


```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of gte-Qwen2-7B-instruct] \
 --dtype=float16 \
 --max-model-len 32768 \
 --block-size=64 \
 --disable-log-requests

# 启动客户端
python -m vllm_utils.benchmark_embedding_rerank \
 --test-type embedding \
 --api-url http://localhost:8000/v1/embeddings \
 --model [path of gte-Qwen2-7B-instruct] \
 --input-len 1024 \
 --total-requests 1 \
 --request-rate inf \
 --tokenizer [path of gte-Qwen2-7B-instruct]
```