## unsloth

### rnj-1-instruct

#### 模型下载
*  url: [rnj-1-instruct](https://modelscope.cn/models/unsloth/rnj-1-instruct)

*  branch: `master`

*  commit id: `57460668`

将上述url设定的路径下的内容全部下载到`rnj-1-instruct`文件夹中。

注：需要安装以下依赖：
#### requirements
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

#### 在线测试
```shell
# 启动服务端
vllm serve "[path of rnj-1-instruct]" \
    --served-model-name rnj \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --block-size=64

# 启动客户端
curl -X POST "http://localhost:8000/v1/chat/completions" \
     -H "Content-Type: application/json"
     --data '{
                "model": "rnj",
                "messages": [
                                {
                                    "role": "user",
                                    "content": "李白是谁"
                                }
                            ]
            }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of rnj-1-instruct]" \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --block-size=64

# 启动客户端
vllm bench serve \
    --backend vllm \
    --dataset-name random \
    --model "[path of rnj-1-instruct]" \
    --num-prompts 256 \
    --max-concurrency 64 \
    --random-input-len 1000 \
    --random-output-len 1000 \
    --trust-remote-code \
    --ignore_eos
```
注：
*  本模型支持的`max-model-len`为32768；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；