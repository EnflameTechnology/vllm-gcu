## llama

### llama-65b

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
* url: [llama-65b](https://github.com/facebookresearch/llama)
* branch: `llama_v1`
* commit id: `57b0eb62de0636e75af471e49e2f1862d908d9d8`

- 参考[download](https://github.com/facebookresearch/llama/tree/llama_v1?tab=readme-ov-file#download)下载llama-65b模型，将全部内容下载到`llama-65b`文件夹内。
- 参考[convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/convert_llama_weights_to_hf.py)，将下载的模型文件转为huggingface transformers格式，将转换的全部内容存放在`llama-65b-hf`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-65b-hf] \
 --demo=te \
 --tensor-parallel-size=4 \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-65b-hf] \
 --tensor-parallel-size=4 \
 --max-model-len=2048 \
 --tokenizer=[path of llama-65b-hf] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama2-13b

#### 模型下载
*  url:[llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf/tree/main)

*  branch:`main`

*  commit id:`638c8be`

将上述url设定的路径下的内容全部下载到`llama-2-13b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-2-13b-hf] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --gpu-memory-utilization=0.945
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-2-13b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of llama-2-13b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.945 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### chinese-llama-2-7b
#### 模型下载
*  url:[chinese-llama-2-7b](https://huggingface.co/hfl/chinese-llama-2-7b)

*  branch:`main`

*  commit id:`c40cf9a`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-7b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-7b-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-7b-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-7b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-7b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### chinese-llama-2-7b-16k
#### 模型下载
*  url:[chinese-llama-2-7b-16k](https://huggingface.co/hfl/chinese-llama-2-7b-16k)

*  branch:`main`

*  commit id:`c934a79`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-7b-16k-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-7b-16k-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-7b-16k-hf] \
 --dtype=float16 \
 --demo=tc \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-7b-16k-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-7b-16k-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16384;

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### chinese-llama-2-13b

#### 模型下载
*  url:[chinese-llama-2-13b](https://huggingface.co/hfl/chinese-llama-2-13b)

*  branch:`main`

*  commit id:`043f8d2`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-13b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-13b-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-13b-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-13b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-13b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### chinese-llama-2-13b-16k

#### 模型下载
*  url:[chinese-llama-2-13b-16k](https://huggingface.co/hfl/chinese-llama-2-13b-16k)

*  branch:`main`

*  commit id:`1c90d65`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-13b-16k-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-13b-16k-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-13b-16k-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-13b-16k-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-13b-16k-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为16384；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Meta-Llama-3.1-405B-Instruct-AWQ-INT4

#### 模型下载
*  url:[Meta-Llama-3.1-405B-Instruct-AWQ-INT4](https://www.modelscope.cn/models/llm-research/meta-llama-3.1-405b-instruct-awq-int4/)

*  branch:`master`

*  commit id:`02d8aeb3`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-405B-Instruct-AWQ-INT4`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3.1-405B-Instruct-AWQ-INT4] \
  --tensor-parallel-size 8 \
  --max-model-len=16384 \
  --output-len=512 \
  --demo=te \
  --dtype=float16 \
  --device gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-405B-Instruct-AWQ-INT4] \
 --tensor-parallel-size 8 \
 --max-model-len=16384 \
 --input-len=14336 \
 --output-len=1024 \
 --dtype=float16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64
```
注：

*  本模型支持的`max-model-len`为131072；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
