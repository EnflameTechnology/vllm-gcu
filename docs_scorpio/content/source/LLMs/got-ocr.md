## GOT-OCR

### GOT-OCR-2.0-hf

#### 模型下载
*  url: [GOT-OCR-2.0-hf](https://modelscope.cn/models/stepfun-ai/GOT-OCR-2.0-hf)

*  branch: `master`

*  commit id: `ec631d19`

将上述url设定的路径下的内容全部下载到`GOT-OCR-2.0-hf`文件夹中。

#### 环境变量

```bash
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
vllm serve "[path of GOT-OCR-2.0-hf]" \
    --tokenizer="[path of GOT-OCR-2.0-hf]" \
    --dtype=bfloat16 \
    --max-model-len=8192 \
    --max-num-batched-tokens=2048 \
    --max-num-seqs=32 \
    --gpu-memory-utilization=0.9 \
    --block-size=64 \
    --tensor-parallel-size=1 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --compilation_config '{"cudagraph_mode":"FULL"}' \
    --port 8080

# 启动客户端
IMAGE_PATH="docstructbench_dianzishu_zhongwenzaixian-o.O-60599898.pdf_30.jpg"
curl -L -O https://raw.githubusercontent.com/opendatalab/OmniDocBench/main/demo_data/omnidocbench_demo/images/$IMAGE_PATH
MODEL="[path of GOT-OCR-2.0-hf]"
VLLM_SERVER="http://localhost:8080"

IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

cat <<EOF | curl -s -X POST "$VLLM_SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- | jq -r '.choices[0].message.content'
{
  "model": "$MODEL",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,$IMAGE_BASE64"
          }
        },
        {
          "type": "text",
          "text": "OCR"
        }
      ]
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.0,
  "top_p": 1.0,
  "stop": ["<|im_end|>", "<|endoftext|>"],
  "stop_token_ids": [151645],
  "repetition_penalty": 1.05
}
EOF
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of GOT-OCR-2.0-hf]" \
    --tokenizer="[path of GOT-OCR-2.0-hf]" \
    --dtype=bfloat16 \
    --max-model-len=8192 \
    --max-num-batched-tokens=2048 \
    --max-num-seqs=32 \
    --gpu-memory-utilization=0.9 \
    --block-size=64 \
    --tensor-parallel-size=1 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --compilation_config '{"cudagraph_mode":"FULL"}' \
    --port 8080


# 启动客户端
vllm bench serve  \
    --model "[path of GOT-OCR-2.0-hf]" \
    --backend openai-chat  \
    --dataset-name random-mm \
    --num-prompts 1  \
    --random-input-len 1024  \
    --random-output-len 1024  \
    --random-mm-base-items-per-request 1 \
    --random-mm-num-mm-items-range-ratio 0.0 \
    --random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --random-mm-bucket-config '{(1024, 1024, 1): 1.0}' \
    --endpoint /v1/chat/completions \
    --port 8080 \
    --trust-remote-code \
    --ignore_eos  \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --metric-percentiles 25,50,75,90,95,99,100
```

注：
*  本模型支持的`max-model-len`为8000；
*  `input-len`、`output-len`和`num-prompts`可按需调整；