## InternVL

### InternVL3_5-30B-A3B

### 模型下载
* url: [InternVL3_5-30B-A3B](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-30B-A3B-Instruct/files)

* branch: `master`

* commit id: `58620057`

将上述url设定的路径下的内容全部下载到`InternVL3_5-30B-A3B`文件夹中。
注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.57.1
```
#### 环境变量

```shell
export VLLM_USE_V1=1
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

#### 在线测试（图像）

* 测试图像：[duck.jpg](https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg)

```shell
# 启动服务端
vllm serve "[path of InternVL3_5-30B-A3B]" \
 --tensor-parallel-size 2 \
 --block-size=64 \
 --async-scheduling \
 --trust-remote-code \
 --no-enable-prefix-caching

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
          "text": "What’s in this picture?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,$(base64 -w0 duck.jpg)"
          }
        }
      ]
    }
  ],
  "model": "[path of InternVL3_5-30B-A3B]",
  "stop": null,
  "stream": false
}'
```

#### 在线测试（视频）

* 测试图像：[ForBiggerFun.mp4](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4)

```shell
# 启动服务端
vllm serve "[path of InternVL3_5-30B-A3B]" \
 --tensor-parallel-size 2 \
 --block-size=64 \
 --async-scheduling \
 --trust-remote-code \
 --no-enable-prefix-caching

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
          "text": "What’s in this video?"
        },
        {
          "type": "video_url",
          "video_url": {
            "url": "data:video/mp4;base64,$(base64 -w0 ForBiggerFun.mp4)"
          }
        }
      ]
    }
  ],
  "model": "[path of InternVL3_5-30B-A3B]",
  "stop": null,
  "stream": false
}'
```


#### 性能测试

```shell
# 启动服务端
vllm serve "[path of InternVL3_5-30B-A3B]" \
 --tensor-parallel-size 2 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --async-scheduling \
 --trust-remote-code


# 启动客户端
evalscope perf \
 --parallel 4 \
 --model "[path of InternVL3_5-30B-A3B]" \
 --url http://localhost:8000/v1/chat/completions \
 --api openai \
 --dataset random_vl \
 --min-tokens 1024 \
 --max-tokens 1024 \
 --prefix-length 0 \
 --min-prompt-length 1024 \
 --max-prompt-length 1024 \
 --image-width 1280 \
 --image-height 720 \
 --image-format RGB \
 --image-num 1 \
 --number 40 \
 --tokenizer-path "[path of InternVL3_5-30B-A3B]" \
 --extra-args '{"ignore_eos": true}'
```

注：
*  测试参数可按需调整；
