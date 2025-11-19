## Qwen-VL

### Qwen2.5-VL-3B-Instruct

#### 模型下载
* url: [Qwen2.5-VL-3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct/files)
* branch: master
* commit id: 1b5a7674

- 将上述url设定的路径下的内容全部下载到`Qwen2.5-VL-3B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
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

#### 批量离线推理
##### 图像推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Image \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test image] \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --disable-async-output-proc \
```
##### 视频推理
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --demo \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Video \
 --prompt=[your prompt] \
 --input-vision-file=[path of your test video] \
 --num-frames 6 \
 --dtype=bfloat16 \
 --max-output-len=128 \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --disable-async-output-proc
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --perf \
 --model=[path of Qwen2.5-VL-3B-Instruct] \
 --model-arch-suffix Image \
 --dtype=bfloat16 \
 --batch-size=1 \
 --input-len=1200 \
 --input-vision-shape="1280,720" \
 --max-output-len=100 \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --trust-remote-code \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --disable-async-output-proc
```
注：
* 默认为graph mode推理，若想使用eager mode，请添加`--enforce-eager`；
* 本模型支持的`max-model-len`为128000；

### Qwen3-VL-30B-A3B-Thinking

#### 模型下载
*  url: [Qwen3-VL-30B-A3B-Thinking](https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking)

*  branch: `master`

*  commit id: `fbe49ed7`

将上述url设定的路径下的内容全部下载到`Qwen3-VL-30B-A3B-Thinking`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
python3 -m pip install evalscope==1.1.0
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

#### serving模式
```shell
# 启动服务器
vllm serve "[path of Qwen3-VL-30B-A3B-Thinking]" \
 --max-model-len 262144 \
 --tensor-parallel-size 4 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code

# 启动客户端
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @- <<EOF
{
  "model": "[path of Qwen3-VL-30B-A3B-Thinking]",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this picture?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,$(base64 -w0 demo.jpeg)"}}
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
vllm serve "[path of Qwen3-VL-30B-A3B-Thinking]" \
 --max-model-len 262144 \
 --tensor-parallel-size 4 \
 --dtype=bfloat16 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --compilation_config '{"cudagraph_mode":"FULL"}' \
 --trust-remote-code


# 启动客户端
evalscope perf \
 --parallel 32 \
 --model "[path of Qwen3-VL-30B-A3B-Thinking]" \
 --url http://127.0.0.1:8000/v1/chat/completions \
 --api openai \
 --dataset random_vl \
 --min-tokens 100 \
 --max-tokens 100 \
 --prefix-length 0 \
 --min-prompt-length 1200 \
 --max-prompt-length 1200 \
 --image-width 1280 \
 --image-height 720 \
 --image-format RGB \
 --image-num 1 \
 --number 64 \
 --tokenizer-path "[path of Qwen3-VL-30B-A3B-Thinking]" \
 --extra-args '{"ignore_eos": true}'
```
注：
*  本模型支持的`max-model-len`为262144；
*  `prompt-length`、`tokens`和`parallel`可按需调整；


