## Yi

### Yi-6B

#### 模型下载
*  url: [Yi-6B](https://huggingface.co/01-ai/Yi-6B/tree/main)

*  branch: `main`

*  commit id: `795e122`

将上述url设定的路径下的内容全部下载到`yi-6b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of yi-6b] \
 --output-len=256 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of yi-6b] \
 --max-model-len=4096 \
 --tokenizer=[path of yi-6b] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Yi-34B

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [Yi-34B](https://huggingface.co/01-ai/Yi-34B/tree/main)

*  branch: `main`

*  commit id: `b65f157`

将上述url设定的路径下的内容全部下载到`yi-34b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of yi-34b] \
 --tensor-parallel-size=2 \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of yi-34b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of yi-34b] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Yi-9B

#### 模型下载
*  url: [Yi-9B](https://huggingface.co/01-ai/Yi-9B/tree/main)

*  branch: `main`

*  commit id: `55e0efc`

将上述url设定的路径下的内容全部下载到`Yi-9B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Yi-9B] \
 --output-len=256 \
 --demo=te \
 --tensor-parallel-size 1 \
 --dtype=bfloat16 \
 --device gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Yi-9B] \
 --max-model-len=4096 \
 --input-len=2048 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16
```

注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Yi-6B-200K

#### 模型下载
*  url: [Yi-6B-200K](https://huggingface.co/01-ai/Yi-6B-200K/tree/main)

*  branch: `main`

*  commit id: `46b2762`

将上述url设定的路径下的内容全部下载到`Yi-6B-200K`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Yi-6B-200K] \
  --tensor-parallel-size 2 \
 --max-model-len=200000 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Yi-6B-200K] \
 --tensor-parallel-size 2 \
 --max-model-len=200000 \
 --input-len=1024 \
 --output-len=1024 \
 --dtype=float16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-coder
```

注：
*  本模型支持的`max-model-len`为200000；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Yi-34B-200K

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [Yi-34B-200K](https://huggingface.co/01-ai/Yi-34B-200K/tree/main)

*  branch: `main`

*  commit id: `09a39628465e62ea2bf199a39ac391135ba59e01`

将上述url设定的路径下的内容全部下载到`yi-34b-200k`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of yi-34b-200k] \
 --tensor-parallel-size=8 \
 --max-model-len=200000 \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --device=gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of yi-34b-200k] \
 --device=gcu \
 --max-model-len=200000 \
 --tokenizer=[path of yi-34b-200k] \
 --dtype=float16 \
 --input-len=198976 \
 --output-len=1024 \
 --num-prompts=1 \
 --tensor-parallel-size=8 \
 --block-size=64
```
注：
*  本模型支持的`max-model-len`为200000；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Yi-1.5-34B-Chat-GPTQ

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载

*  url: [Yi-1.5-34B-Chat-GPTQ](https://www.modelscope.cn/models/AI-ModelScope/Yi-1.5-34B-Chat-GPTQ/files)

*  branch: `master`

*  commit id: `97535d73`

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo="te" \
    --model [path of Yi-1.5-34B-Chat-GPTQ] \
    --output-len=256 \
    --dtype=float16 \
    --tensor-parallel-size 2 \
    --device gcu \
    --max-model-len=4096
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
    --model [path of Yi-1.5-34B-Chat-GPTQ] \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --quantization gptq \
    --trust-remote-code \
    --num-prompts 8 \
    --max-model-len 4096 \
    --input-len 512 \
    --output-len 512 \
    --device gcu \
    --block-size=64
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

