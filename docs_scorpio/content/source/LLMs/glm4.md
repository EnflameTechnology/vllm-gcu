## GLM4
### GLM-Z1-32B-0414-GPTQ-Int4

#### 模型下载
*  url: [GLM-Z1-32B-0414-GPTQ-Int4](https://www.modelscope.cn/models/tclf90/glm-z1-32b-0414-gptq-int4/files)

*  branch: `master`

*  commit id: `ec6ecf2793f061005011f078e8ae0975bcb5ace8`

将上述url设定的路径下的内容全部下载到`GLM-Z1-32B-0414-GPTQ-Int4`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.53.2
python3 -m pip install triton==3.1.0
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

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of GLM-Z1-32B-0414-GPTQ-Int4] \
 --demo=te \
 --dtype=bfloat16 \
 --quantization=gptq_gcu \
 --tensor-parallel-size=1 \
 --output-len=128 \
 --block-size=64 \
 --max-model-len=32768 \
 --gpu-memory-utilization=0.9 \
 --trust-remote-code \
 --device gcu \
 --disable-async-output-proc
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of GLM-Z1-32B-0414-GPTQ-Int4] \
  --num-scheduler-steps=16 \
  --tensor-parallel-size 1 \
  --max-seq-len-to-capture=32768 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --quantization=gptq_gcu \
  --disable-async-output-proc


# 启动客户端
  python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of GLM-Z1-32B-0414-GPTQ-Int4] \
  --num-prompts 1 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos \
  --strict-in-out-len \
  --keep-special-tokens
```

注：
*  GLM-Z1-32B-0414-GPTQ-Int4模型支持的`max-model-len`为32k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
