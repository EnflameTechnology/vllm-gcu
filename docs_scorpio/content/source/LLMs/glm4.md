## GLM4
### GLM-Z1-32B-0414-GPTQ-Int4

#### 模型下载
*  url: [GLM-Z1-32B-0414-GPTQ-Int4](https://www.modelscope.cn/models/tclf90/glm-z1-32b-0414-gptq-int4/files)

*  branch: `master`

*  commit id: `ec6ecf2793f061005011f078e8ae0975bcb5ace8`

将上述url设定的路径下的内容全部下载到`GLM-Z1-32B-0414-GPTQ-Int4`文件夹中。

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

#### 在线测试

```shell
# 启动服务器
vllm serve "[path of GLM-Z1-32B-0414-GPTQ-Int4]" \
 --dtype=bfloat16 \
 --quantization=moe_wna16_gcu \
 --tensor-parallel-size=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --trust-remote-code 
 
# 客户端
curl "http://127.0.0.1:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{"max_tokens": 500,"messages": [{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content":"李白是谁？"}],"model":"/ard-data/pretrained_models/glm-z1-32b-0414-gptq-int4/","stop": null,"stream": false}'

```

#### 性能测试

```shell
# 启动服务端
  vllm serve "[path of GLM-Z1-32B-0414-GPTQ-Int4]" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --quantization=moe_wna16_gcu \
  --no-enable-prefix-caching


# 启动客户端
vllm bench serve --model "[path of GLM-Z1-32B-0414-GPTQ-Int4]" \
  --backend vllm \
  --dataset-name random \
  --num-prompts 1 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos 
```

注：
*  GLM-Z1-32B-0414-GPTQ-Int4模型支持的`max-model-len`为32k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

