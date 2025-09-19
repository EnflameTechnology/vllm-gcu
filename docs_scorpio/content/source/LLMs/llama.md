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
 --model=[path of Meta-Llama-3.1-8B-Instruct] \
 --demo=te \
 --dtype=bfloat16 \
 --output-len=256 \
 --device=gcu \
 --max-model-len=32768 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9 \
 --disable-async-output-proc
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-8B-Instruct] \
 --max-model-len=32768 \
 --tokenizer=[path of Meta-Llama-3.1-8B-Instruct] \
 --input-len=8192 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为131072, 单张卡可跑32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
