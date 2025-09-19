## DeepSeek
### DeepSeek-R1-Distill-Qwen-7B

#### 模型下载
*  url: [DeepSeek-R1-Distill-Qwen-7B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

*  branch: `master`

*  commit id: `6bf9b8f2`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1-Distill-Qwen-7B`文件夹中。
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
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code \
 --disable-async-output-proc
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server  \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --tensor-parallel-size 1 \
 --max-model-len 32768 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --disable-async-output-proc


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --model [path of DeepSeek-R1-Distill-Qwen-7B] \
 --backend vllm \
 --dataset-name random \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
