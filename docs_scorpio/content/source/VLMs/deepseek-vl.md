## deepseek-vl

### deepseek-vl2

#### 模型下载
* url: [deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2/tree/main)
* branch: main
* commit id: f363772d1c47f4239dd844015b4bd53beb87951b

- 将上述url设定的路径下的内容全部下载到`deepseek-vl2`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install timm==1.0.15
```
#### 环境变量

```
export VLLM_USE_V1=0
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
```

#### 在线推理
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --limit-mm-per-prompt image=1 \
 --model [path of deepseek-vl2] \
 --max-model-len 4096 \
 --block-size 64 \
 --hf-overrides='{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
 --dtype=bfloat16 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code \
 --seed 0 \
 --tensor-parallel-size 2 \
 --allowed-local-media-path=[absolute path of your image folder] \
 --served-model-name=deepseek-vl2

# 启动客户端
curl "http://0.0.0.0:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": [your prompt]
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file://[absolute path of your image]"
            }
          }
        ]
      }
    ],
    "model": deepseek-vl2,
    "temperature": 0,
    "top_p": 0.01,
    "repetition_penalty": 1.05,
    "stop": null,
    "stream": false
  }'
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* `--allowed-local-media-path`请设置为一个绝对路径，其中包含推理所需的图片文件

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_vision_language --perf \
 --model=[path of deepseek-vl2] \
 --mm-per-prompt=1 \
 --max-model-len=4096 \
 --tensor-parallel-size=2 \
 --dtype=bfloat16 \
 --input-vision-shape="1024,1024" \
 --gpu-memory-utilization=0.9 \
 --block-size=64 \
 --input-len=2048 \
 --max-output-len=2048 \
 --batch-size=1 \
 --trust-remote-code \
 --device=gcu \
 --hf-overrides='{"architectures": ["DeepseekVLV2ForCausalLM"]}' \
 --repetition-penalty=1.05 \
 --top_p=0.01
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为4096；