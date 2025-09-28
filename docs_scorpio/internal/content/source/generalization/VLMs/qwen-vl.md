## Qwen-VL



### Qwen2.5-VL-3B-Instruct

#### 模型下载
* url: [Qwen2.5-VL-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/files)
* branch: `master`
* commit id: `1b5a7674`

- 将上述url设定的路径下的内容全部下载到`Qwen2.5-VL-3B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.52.3 opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86 evalscope 'evalscope[perf]'
```

#### 环境变量

```shell
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 离线推理

```shell
# 启动服务端
vllm serve "[path of Qwen2.5-VL-3B-Instruct]" \
 --max-model-len 32768 \
 --block-size 64 \
 --dtype=bfloat16 \
 --trust-remote-code \
 --seed 0 \
 --tensor-parallel-size 1 \
 --device gcu

#启动客户端
curl "http://localhost:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "max_tokens": 500,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "[your prompt]"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "[url of your test image]"
          }
        }
      ]
    }
  ],
  "model": "[path of Qwen2.5-VL-3B-Instruct]",
  "stop": null,
  "stream": false
}'

```

#### 性能测试
```shell
# 启动服务端
vllm serve "[path of Qwen2.5-VL-3B-Instruct]" \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --limit-mm-per-prompt image=16

# 启动客户端
evalscope perf \
 --parallel 1 \
 --model "[path of Qwen2.5-VL-3B-Instruct]" \
 --url http://localhost:8000/v1/chat/completions \
 --api openai \
 --dataset random_vl \
 --min-tokens 100 \
 --max-tokens 100 \
 --prefix-length 0 \
 --min-prompt-length 1200 \
 --max-prompt-length 1200 \
 --image-width 512 \
 --image-height 512 \
 --image-format RGB \
 --image-num 1 \
 --number 100 \
 --tokenizer-path "[path of Qwen2.5-VL-3B-Instruct]" \
 --debug
```

