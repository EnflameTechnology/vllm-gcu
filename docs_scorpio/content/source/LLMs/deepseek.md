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
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### DeepSeek-R1-Distill-Llama-8B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Llama-8B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/)

*  branch: `master`

*  commit id: `b1a59cb3`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Llama-8B`文件夹中。

#### 环境变量

```
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试
```shell
# 启动服务器
vllm serve "[path of DeepSeek-R1-Distill-Llama-8B]" \
 --max-model-len 32768 \
 --tensor-parallel-size 1 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code


# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of DeepSeek-R1-Distill-Llama-8B]",
        "prompt": [
          "请介绍北京的旅游景点",
          "介绍一下大熊猫",
          "晚上睡不着应该怎么办",
          "李白的代表作有哪些？"
        ],
        "max_tokens": 128,
        "temperature": 0
     }'

```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of DeepSeek-R1-Distill-Llama-8B]" \
 --max-model-len 32768 \
 --tensor-parallel-size 1 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code


# 启动客户端
vllm bench serve --model "[path of DeepSeek-R1-Distill-Llama-8B]" \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 2 \
 --max-concurrency 1 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl \
 --metric-percentiles 25,50,75,90,99,100
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### DeepSeek-R1-Distill-Llama-70B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/)

*  branch: `master`

*  commit id: `c298a156`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Llama-70B`文件夹中。

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
vllm serve "[path of DeepSeek-R1-Distill-Llama-70B]" \
 --tokenizer=[path of DeepSeek-R1-Distill-Llama-70B] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --trust_remote_code


# 启动客户端
curl "http://127.0.0.1:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{"max_tokens": 500,"messages": [{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content":"李白是谁？"}],"model":"/ard-data/pretrained_models/deepseek-r1-distill-llama-70b/","stop": null,"stream": false}'

```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of DeepSeek-R1-Distill-Llama-70B]" \
 --tokenizer=[path of DeepSeek-R1-Distill-Llama-70B] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --trust-remote-code


# 启动客户端
vllm bench serve \
 --backend vllm \
 --dataset-name random \
 --model [path of DeepSeek-R1-Distill-Llama-70B] \
 --max-concurrency 1 \
 --num-prompts 16 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
