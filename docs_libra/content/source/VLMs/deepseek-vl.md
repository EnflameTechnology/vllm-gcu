## deepseek-vl

### DeepSeek-OCR

#### 模型下载
* url: [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR/)
* branch: main
* commit id: 2c968b4

- 将上述url设定的路径下的内容全部下载到`DeepSeek-OCR`文件夹中。

### 测试图片下载
* url: [show1.jpg](https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/assets/show1.jpg)

#### 环境变量

```
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线推理
```shell
# 启动服务端
vllm serve [path of DeepSeek-OCR]  \
 --max-model-len 8192 \
 --tensor-parallel-size 1 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code \
 --mm-processor-cache-gb 0 \
 --hf-overrides '{"model_type":"deepseek_ocr"}'

# 启动客户端
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "[path of DeepSeek-OCR]",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Free OCR. "},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$(base64 -w0 show1.jpg)"}}
      ]
    }
  ],
  "max_tokens": 256
}
EOF
```

#### 性能测试

```shell
# 启动服务端
vllm serve [path of DeepSeek-OCR] \
 --max-model-len 8192 \
 --tensor-parallel-size 1 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code \
 --mm-processor-cache-gb 0 \
 --hf-overrides '{"model_type":"deepseek_ocr"}'


# 启动客户端
vllm bench serve \
 --backend openai-chat \
 --endpoint /v1/chat/completions \
 --model [path of DeepSeek-OCR] \
 --dataset-name random-mm \
 --random-input-len 10 \
 --random-output-len 512 \
 --random-prefix-len 0 \
 --random-mm-base-items-per-request 1 \
 --random-mm-bucket-config '{(512,512,1):1}' \
 --num-prompts 320 \
 --max-concurrency 32 \
 --ignore-eos \
 --percentile-metrics ttft,tpot,itl,e2el \
 --metric-percentiles 25,50,75,90,95,99,100

```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为8192；

