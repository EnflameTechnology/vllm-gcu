## OpenBMB
### MiniCPM4.1-8B

#### 模型下载
*  url: [MiniCPM4.1-8B](https://modelscope.cn/models/OpenBMB/MiniCPM4.1-8B/files)

*  branch: `master`

*  commit id: `f03f38e9`

将上述url设定的路径下的内容全部下载到`MiniCPM4.1-8B`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
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
# 启动服务器
vllm serve "[path of MiniCPM4.1-8B]" \
  --served-model-name MiniCPM4.1-8B \
  --hf-overrides='{"max_position_embeddings": 131072}' \
  --tensor-parallel-size 1 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --trust-remote-code
 
# 启动客户端
curl "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
        "max_tokens": 500,
        "messages": [
                        {
                            "role": "system",
                             "content": "You are a helpful assistant."
                         },
                         {
                             "role":"user",
                             "content":"李白是谁？"
                         }
                     ],
        "model":"MiniCPM4.1-8B",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of MiniCPM4.1-8B]" \
  --hf-overrides='{"max_position_embeddings": 131072}' \
  --tensor-parallel-size 1 \
  --block-size=64 \
  --no-enable-prefix-caching \
  --trust-remote-code


# 启动客户端
vllm bench serve --model "[path of MiniCPM4.1-8B]" \
  --backend vllm \
  --dataset-name random \
  --num-prompts 2 \
  --max-concurrency 1 \
  --random-input-len 1024 \
  --random-output-len 130000 \
  --trust-remote-code \
  --ignore_eos \
  --percentile-metrics ttft,tpot,itl \
  --metric-percentiles 25,50,75,90,99,100
```

注：
*  MiniCPM4.1-8B模型支持的`max-model-len`默认为64k，通过--hf-overrides='{"max_position_embeddings": 131072}'可拓展到128k；
*  测试参数可按需调整；