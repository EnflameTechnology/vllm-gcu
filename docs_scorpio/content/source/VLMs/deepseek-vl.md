## deepseek-vl

### deepseek-vl2

#### 模型下载
* url: [deepseek-vl2](https://modelscope.cn/models/deepseek-ai/deepseek-vl2/)
* branch: main
* commit id: 9ad9b50e

- 将上述url设定的路径下的内容全部下载到`deepseek-vl2`文件夹中。
- 将模型chat_template.jinja 放入模型目录

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
export VLLM_GCU_DEEPSEEK_FUSION=0
```

#### 在线推理
```shell
# 启动服务端
vllm serve [path of deepseek-vl2] \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --limit-mm-per-prompt '{"image": 6}' \
 --hf_overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
 -tp 2 \
 --max-model-len 4096 \
 --block-size 64 \
 --trust_remote_code \
 --mm-processor-cache-gb 0 \
 --seed 0 \
 --no-enable-prefix-caching

# 启动客户端
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "[path of deepseek-vl2]",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "[prompt of test]"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$(base64 -w0 image.png)"}}
      ]
    }
  ],
  "max_tokens": 2560,
  "temperature": 0, "top_p": 0.01, "repetition_penalty": 1.05, "stop": null,"stream": false
}
EOF
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 提示词模板下载地址为[chat_template.jinja](https://github.com/vllm-project/vllm/blob/v0.8.0/examples/template_deepseek_vl2.jinja)；

#### 性能测试

```shell
# 启动服务端
vllm serve [path of deepseek-vl2] \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --limit-mm-per-prompt '{"image": 6}' \
 --hf_overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
 -tp 2 \
 --max-model-len 4096 \
 --block-size 64 \
 --trust_remote_code \
 --mm-processor-cache-gb 0 \
 --seed 0 \
 --no-enable-prefix-caching


# 启动客户端
evalscope perf \
 --parallel 1 \
 --model [path of deepseek-vl2] \
 --tokenizer-path [tokenizer of deepseek-vl2] \
 --url http://127.0.0.1:8000/v1/chat/completions \
 --api openai \
 --dataset random_vl \
 --min-tokens 2048 \
 --max-tokens 2048 \
 --prefix-length 0 \
 --min-prompt-length 256 \
 --max-prompt-length 256 \
 --image-width 1024 \
 --image-height 1024 \
 --image-format RGB \
 --image-num 1 \
 --number 10 \
 --extra-args '{"ignore_eos": true}' \
 --repetition-penalty 1.05 \
 --top-p 0.01 \
 --temperature 0.0

```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为4096；