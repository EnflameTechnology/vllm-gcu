## llama
### Meta-Llama-3.1-8B-Instruct

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main/)

*  branch: `main`

*  commit id: `8c22764`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-8B-Instruct`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
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
# 服务端
vllm serve "[path of Meta-Llama-3.1-8B-Instruct]" \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --block-size=64 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --no-enable-prefix-caching \
 --trust_remote_code 

# 客户端
curl "http://127.0.0.1:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{"max_tokens": 500,"messages": [{"role": "system", "content": "You are a helpful assistant."},{"role":"user","content":"李白是谁？"}],"model":"/ard-data/pretrained_models/Meta-Llama-3.1-8B-Instruct/","stop": null,"stream": false}'
```

#### 性能测试

```shell

# 服务端
vllm serve "[path of Meta-Llama-3.1-8B-Instruct]" \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --no-enable-prefix-caching

 # 客户端
vllm bench serve --model [path of Meta-Llama-3.1-8B-Instruct] \
  --backend vllm \
  --dataset-name random \
  --num-prompts 10 \
  --max-concurrency 1 \
  --random-input-len 8192 \
  --random-output-len 512 \
  --trust-remote-code \
  --ignore_eos 
```
注：
*  本模型支持的`max-model-len`为131072, 单张卡可跑32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；