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

### GLM-4.5-Air

#### 模型下载
*  url: [GLM-4.5-Air](https://www.modelscope.cn/models/ZhipuAI/GLM-4.5-Air)

*  branch: `master`

*  commit id: `5d60b0e3`

将上述url设定的路径下的内容全部下载到`GLM-4.5-Air`文件夹中。

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
vllm serve "[path of GLM-4.5-Air]" \
 --tokenizer="[path of GLM-4.5-Air]"  \
 --dtype=bfloat16 \
 --max-model-len=131072 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --disable-log-requests \
 --gpu-memory-utilization=0.9 \
 --trust-remote-code \
 --no-enable-prefix-caching

# 启动客户端
curl "http://127.0.0.1:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
    "max_tokens": 500,
    "prompt":["请介绍北京的旅游景点"],
    "model": "[path of GLM-4.5-Air]" ,
    "stop": null,
    "stream": false
    }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of GLM-4.5-Air]" \
 --tokenizer=GLM-4.5-Air \
 --dtype=bfloat16 \
 --max-model-len=131072 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --no-enable-prefix-caching \
 --trust-remote-code


# 启动客户端
vllm bench serve --model "[path of GLM-4.5-Air]"  \
 --base-url http://127.0.0.1:8000 \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 80 \
 --max-concurrency 8 \
 --trust-remote-code \
 --ignore-eos \
 --percentile-metrics ttft,tpot,itl,e2el \
 --metric-percentiles 25,50,75,90,99,100
```

注：
*  GLM-4.5-Air模型支持的`max-model-len`为128k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### GLM-4.5-Air-PAD_gptq_w4a16

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`GLM-4.5-Air-PAD_gptq_w4a16.tar`文件并解压，将压缩包内的内容全部拷贝到`GLM-4.5-Air-PAD_gptq_w4a16`文件夹中。
- `GLM-4.5-Air-PAD_gptq_w4a16`目录结构如下所示：

```shell
GLM-4.5-Air-PAD_gptq_w4a16
├── README.md
├── chat_template.jinja
├── config.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── tops_quantize_info.json
```

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
# 启动服务器
vllm serve "[path of GLM-4.5-Air-PAD_gptq_w4a16]" \
 --max-model-len 131072 \
 --tensor-parallel-size 4 \
 --block-size 64 \
 --no-enable-prefix-caching \
 --trust-remote-code \
 --quantization moe_wna16_gcu

# 启动客户端
curl "http://127.0.0.1:8000/v1/completions" \
-H "Content-Type: application/json" \
-d '{
    "max_tokens": 500,
    "prompt":["请介绍北京的旅游景点"],
    "model": "[path of GLM-4.5-Air-PAD_gptq_w4a16]" ,
    "stop": null,
    "stream": false
    }'
```

#### 性能测试

```shell
# 启动服务端
vllm serve "[path of GLM-4.5-Air-PAD_gptq_w4a16]" \
 --max-model-len=131072 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --no-enable-prefix-caching \
 --trust-remote-code \
 --quantization moe_wna16_gcu

# 启动客户端
vllm bench serve --model "[path of GLM-4.5-Air-PAD_gptq_w4a16]"  \
 --base-url http://127.0.0.1:8000 \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 80 \
 --max-concurrency 8 \
 --trust-remote-code \
 --ignore-eos \
 --percentile-metrics ttft,tpot,itl,e2el \
 --metric-percentiles 25,50,75,90,99,100
```

注：
*  GLM-4.5-Air-PAD_gptq_w4a16模型支持的`max-model-len`为128k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；