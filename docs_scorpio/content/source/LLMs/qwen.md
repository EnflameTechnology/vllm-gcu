## Qwen

### Qwen3-32B

#### 模型下载
*  url: [Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)

*  branch: `master`

*  commit id: `bc4962f6`

将上述url设定的路径下的内容全部下载到`Qwen3-32B`文件夹中。

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
 --model [path of Qwen3-32B] \
 --tensor-parallel-size 4 \
 --max-model-len=40960 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code \
 --disable-async-output-proc
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen3-32B] \
 --tensor-parallel-size 4 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefill \
 --disable-async-output-proc


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-32B] \
 --num-prompts 32 \
 --random-input-len 1000 \
 --random-output-len 700 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### QWen3-30B-A3B

#### 模型下载
*  url: [QWen3-30B-A3B](https://www.modelscope.cn/models/Qwen/QWen3-30B-A3B/files)

*  branch: `master`

*  commit id: `e34b3e98`

将上述url设定的路径下的内容全部下载到`QWen3-30B-A3B`文件夹中。

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
  --model [path of QWen3-30B-A3B] \
  --tensor-parallel-size=2 \
  --max-model-len=32768 \
  --output-len=128 \
  --demo=te \
  --dtype=bfloat16 \
  --device gcu \
  --disable-async-output-proc
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of QWen3-30B-A3B] \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --disable-log-requests \
  --block-size=64 \
  --dtype=bfloat16 \
  --device gcu \
  --trust-remote-code \
  --gpu-memory-utilization=0.9 \
  --enable-chunked-prefill \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
  --disable-async-output-proc


# 启动客户端
 python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of QWen3-30B-A3B] \
  --num-prompts 1 \
  --random-input-len 130048 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos \
  --strict-in-out-len \
  --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `random-input-len`、`random-output-len`和`num-prompts`可按需调整；

### QwQ-32B
#### 模型下载
*  url: [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/files)

*  branch: `master`

*  commit id: `887ddbd72be5bed61cd702b87cd2fc25e65a708d`

将上述url设定的路径下的内容全部下载到`QwQ-32B`文件夹中。
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
 python3.10 -m vllm_utils.benchmark_test \
 --model=[path of QwQ-32B] \
 --demo=te \
 --tensor-parallel-size 4 \
 --max-model-len=32768 \
 --output-len=1024 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code \
 --disable-async-output-proc
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of QwQ-32B] \
  --num-scheduler-steps=16 \
  --tensor-parallel-size 4 \
  --max-seq-len-to-capture=32768 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=bfloat16 \
  --disable-async-output-proc


# 启动客户端
  python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of QwQ-32B] \
  --num-prompts 1 \
  --random-input-len 512 \
  --random-output-len 512 \
  --trust-remote-code \
  --ignore_eos \
  --strict-in-out-len \
  --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

### Qwen3-Next-80B-A3B-Instruct

#### 模型下载
*  url: [Qwen3-Next-80B-A3B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct/)

*  branch: `master`

*  commit id: `34fec46d`

将上述url设定的路径下的内容全部下载到`Qwen3-Next-80B-A3B-Instruct`文件夹中。

#### 环境变量

```
export TORCHGCU_INDUCTOR_ENABLE=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
```

#### serving模式
```shell
# 启动服务器
vllm serve  "[path of Qwen3-Next-80B-A3B-Instruct]" \
	--tensor-parallel-size 8 \
	--dtype=bfloat16 \
	--trust-remote-code \
	--block-size=256 \
	--max-model-len=262144 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--compilation_config '{"cudagraph_mode":"FULL"}' \
	--gpu-memory-utilization 0.8

# 启动客户端
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "[path of Qwen3-Next-80B-A3B-Instruct]",
        "prompt": [
          "请介绍北京的旅游景点",
          "介绍一下大熊猫",
          "晚上睡不着应该怎么办",
          "李白的代表作有哪些？"
        ],
        "max_tokens": 128,
        "temperature": 0
     }'


```

#### 性能测试

```shell
# 启动服务端
vllm serve  "[path of Qwen3-Next-80B-A3B-Instruct]" \
	--tensor-parallel-size 8 \
	--dtype=bfloat16 \
	--trust-remote-code \
	--block-size=256 \
	--max-model-len=262144 \
	--no-enable-prefix-caching \
	--async-scheduling \
	--compilation_config '{"cudagraph_mode":"FULL"}' \
	--gpu-memory-utilization 0.8


# 启动客户端
vllm bench serve --model "[path of Qwen3-Next-80B-A3B-Instruct]" \
 --dataset-name random \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --num-prompts 4 \
 --max-concurrency 1 \
 --trust-remote-code \
 --save-result \
 --ignore-eos \
 --result-filename serving_result.json \
 --percentile-metrics ttft,tpot,itl \
 --metric-percentiles 25,50,75,90,99,100
```
注：
*  本模型支持的`max-model-len`为262144；
*  `input-len`、`output-len`和`num-prompts`可按需调整；

