## gemma

### gemma-2b

#### 模型下载
*  url: [gemma-2b](https://huggingface.co/google/gemma-2b/tree/main)

*  branch: `main`

*  commit id: `9d067f0`

将上述url设定的路径下的内容全部下载到`gemma-2b`文件夹中。

#### requirement
* vllm >= 0.3.3

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of gemma-2b] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of gemma-2b] \
 --tokenizer=[path of gemma-2b] \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codegemma-7b

#### 模型下载
*  url: [codegemma-7b](https://huggingface.co/google/codegemma-7b)

*  branch: `main`

*  commit id: `2ec9700`

将上述url设定的路径下的内容全部下载到`codegemma-7b`文件夹中。

#### requirement
* vllm >= 0.3.3

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codegemma-7b] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --max-model-len=8192
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codegemma-7b] \
 --tokenizer=[path of codegemma-7b] \
 --max-model-len=8192 \
 --input-len=4096 \
 --output-len=4096 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### gemma-7b

#### 模型下载
*  url: [gemma-7b](https://huggingface.co/google/gemma-7b/tree/main)

*  branch: `main`

*  commit id: `359f554`

将上述url设定的路径下的内容全部下载到`gemma-7b`文件夹中。

#### requirement
* vllm >= 0.3.3

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of gemma-7b] \
 --demo=te \
 --dtype=float16 \
 --output-len=20 \
 --max-model-len=8192
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of gemma-7b] \
 --tokenizer=[path of gemma-7b] \
 --max-model-len=8192 \
 --input-len=512 \
 --output-len=7680 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;