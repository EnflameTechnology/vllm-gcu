## MiniCPM-V

### MiniCPM-V-4.5

#### 模型下载
* url: [MiniCPM-V-4_5](https://huggingface.co/openbmb/MiniCPM-V-4_5/)
* branch: `main`
* commit id: `c1a6986b`

- 将上述url设定的路径下的内容全部下载到`MiniCPM-V-4.5`文件夹中。

### 测试图片下载
* url: [test_img.jpg](https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg)

#### requirements

```shell
python3 -m pip install timm==1.0.22
python3 -m pip install evalscope==1.1.0
```

#### 环境变量

```
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
```

#### 在线推理
```shell
# 启动服务端
vllm serve [path of MiniCPM-V-4.5] \
 --block-size 64 \
 --trust-remote-code \

# 启动客户端
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "[path of MiniCPM-V-4.5]",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "[prompt of test]"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$(base64 -w0 [path of test_img.jpg])"}}
      ]
    }
  ],
  "max_tokens": 500,
  "stop": null,"stream": false
}
EOF
```

#### 性能测试

```shell
# 启动服务端
vllm serve [path of MiniCPM-V-4.5] \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --port 8989 \
 --trust-remote-code \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}'

# 启动客户端
evalscope perf \
 --parallel 20 \
 --model [path of MiniCPM-V-4.5] \
 --url http://127.0.0.1:8989/v1/chat/completions \
 --api openai \
 --dataset random_vl \
 --min-tokens 128 \
 --max-tokens 128 \
 --prefix-length 0 \
 --min-prompt-length 100 \
 --max-prompt-length 100 \
 --image-width 512 \
 --image-height 512 \
 --image-format RGB \
 --image-num 1 \
 --number 100 \
 --tokenizer-path [tokenizer of MiniCPM-V-4.5]
```