## DeepSeek
### DeepSeek-R1-Distill-Qwen-7B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

*  branch: `master`

*  commit id: `6bf9b8f2`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-7B`文件夹中。
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

#### 离线模式
```shell
# 启动服务端
vllm serve "[path of DeepSeek-R1-Distill-Qwen-7B]" \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --dtype=bfloat16 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code \
 --block-size=64 \
 --no-enable-prefix-caching

# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" \
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
        "model":"[path of DeepSeek-R1-Distill-Qwen-7B]",
        "stop": null,
        "stream": false
      }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of DeepSeek-R1-Distill-Qwen-7B]" \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --no-enable-prefix-caching


# 启动客户端
vllm bench serve \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 10 \
 --max-concurrency 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
