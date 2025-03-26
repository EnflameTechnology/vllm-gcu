## vicuna
### vicuna-13b-v1.5
#### 模型下载
*  url:[vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5/tree/main)

*  branch:`main`

*  commit id:`3deb010`

将上述url设定的路径下的内容全部下载到`lmsys_vicuna-13b-v1.5`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of lmsys_vicuna-13b-v1.5] \
 --output-len=16 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of lmsys_vicuna-13b-v1.5] \
 --max-model-len=4096 \
 --tokenizer=[path of lmsys_vicuna-13b-v1.5] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=2 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency；

### vicuna-13b-v1.5-16k
#### 模型下载
*  url:[vicuna-13b-v1.5-16k](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/tree/main)

*  branch:`main`

*  commit id:`17c61f9`

将上述url设定的路径下的内容全部下载到`vicuna-13b-v1.5-16k`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of vicuna-13b-v1.5-16k] \
 --max-model-len=12288 \
 --output-len=256 \
 --demo=te \
 --dtype=float16
```
注：
*  本模型在ecc off模式下单卡支持的`max-model-len`为16k，ecc on模式下单卡支持的`max-model-len`为12k，使能16k需要两卡；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of vicuna-13b-v1.5-16k] \
 --max-model-len=12288 \
 --tokenizer=[path of vicuna-13b-v1.5-16k] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下单卡支持的`max-model-len`为16k，ecc on模式下单卡支持的`max-model-len`为12k，使能16k需要两卡；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency；

### vicuna-33b-v1.3

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url:[vicuna-33b-v1.3](https://huggingface.co/lmsys/vicuna-33b-v1.3/tree/main)

*  branch:`main`

*  commit id:`ef8d6be`

将上述url设定的路径下的内容全部下载到`vicuna-33b-v1.3`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of vicuna-33b-v1.3] \
 --tensor-parallel-size=2 \
 --output-len=20 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of vicuna-33b-v1.3] \
 --tensor-parallel-size=2 \
 --max-model-len=2048 \
 --tokenizer=[path of vicuna-33b-v1.3] \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=4 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为2048；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency；
