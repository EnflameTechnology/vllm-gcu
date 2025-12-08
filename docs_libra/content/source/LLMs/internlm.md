## internlm

### internlm3-8b-instruct

#### 模型下载
*  url: [internlm3-8b-instruct](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm3-8b-instruct)

*  branch: `master`

*  commit id: `03ffaab0`

将上述 url 路径下的内容全部下载到 `internlm3-8b-instruct` 文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.55.2
```

#### 环境变量
```
# v1 engine

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TORCHGCU_INDUCTOR_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TOPS_VISIBLE_DEVICES=0
```

#### 在线测试


```shell
# 启动服务端

vllm serve [path of internlm3-8b-instruct] \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --disable-log-requests \
        --block-size=64 \
        --dtype=bfloat16 \
        --port 8003 \
        --served-model-name internlm3-8b-instruct \
        --tokenizer [path of internlm3-8b-instruct] \
        --trust-remote-code \
        --gpu-memory-utilization=0.95 \
        --no-enable-prefix-caching \
        --async-scheduling


# 启动客户端

curl "http://localhost:8003/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"max_tokens": 50,
       "messages": [{"role":"user","content":"What is Deep Learning?"}],
       "model": "internlm3-8b-instruct",
       "stream": false}'

```


#### 性能测试

```shell
# 启动服务端
vllm serve [path of internlm3-8b-instruct] \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --disable-log-requests \
        --block-size=64 \
        --dtype=bfloat16 \
        --port 8003 \
        --served-model-name internlm3-8b-instruct \
        --tokenizer [path of internlm3-8b-instruct] \
        --trust-remote-code \
        --gpu-memory-utilization=0.95 \
        --no-enable-prefix-caching \
        --async-scheduling

# 启动客户端
vllm bench serve \
        --backend vllm  \
        --base-url "http://localhost:8003" \
        --dataset-name random  \
        --model internlm3-8b-instruct \
        --tokenizer [path of internlm3-8b-instruct] \
        --num-prompts 10 \
        --max-concurrency 1 \
        --random-input-len 2048 \
        --random-output-len 1024 \
        --trust-remote-code \
        --ignore_eos
```
