## xuanyuan

### XuanYuan-6B
#### 模型下载
*  url:[XuanYuan-6B](https://www.modelscope.cn/models/Duxiaoman-DI/XuanYuan-6B/files)

*  branch:`master`

*  commit id:`3455536233af69f9d4bad38af80789acf44df291`

将上述url设定的路径下的内容全部下载到`XuanYuan-6B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of XuanYuan-6B] \
 --output-len=20 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of XuanYuan-6B] \
 --max-model-len=8192 \
 --tokenizer=[path of XuanYuan-6B] \
 --input-len=1024 \
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


### XuanYuan-13B
#### 模型下载
*  url:[XuanYuan-13B](https://www.modelscope.cn/models/Duxiaoman-DI/XuanYuan-13B/files)

*  branch:`master`

*  commit id:`9aeb8a5072ee51cd1195610cf93228915c54d268`

将上述url设定的路径下的内容全部下载到`XuanYuan-13B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of XuanYuan-13B] \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=8192
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of XuanYuan-13B] \
 --max-model-len=8192 \
 --tokenizer=[path of XuanYuan-13B] \
 --input-len=1024 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### XuanYuan-70B
#### 模型下载
*  url:[XuanYuan-70B](https://modelscope.cn/models/Duxiaoman-DI/XuanYuan-70B/files)

*  branch:`master`

*  commit id:`fa069fe6`

将上述url设定的路径下的内容全部下载到`XuanYuan-70B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of XuanYuan-70B] \
 --tensor-parallel-size=4 \
 --output-len=256 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of XuanYuan-70B] \
 --tensor-parallel-size=4 \
 --max-model-len=8192 \
 --tokenizer=[path of XuanYuan-70B] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### XuanYuan-70B-Chat
#### 模型下载
*  url:[XuanYuan-70B-Chat](https://modelscope.cn/models/Duxiaoman-DI/XuanYuan-70B-Chat/files)

*  branch:`master`

*  commit id:`59885cde`

将上述url设定的路径下的内容全部下载到`XuanYuan-70B-Chat`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of XuanYuan-70B-Chat] \
 --tensor-parallel-size=4 \
 --output-len=256 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of XuanYuan-70B-Chat] \
 --tensor-parallel-size=4 \
 --max-model-len=8192 \
 --tokenizer=[path of XuanYuan-70B-Chat] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### XuanYuan2-70B

本模型推理及性能测试需要4张enflame gcu。
#### 模型下载
* url:[XuanYuan2-70B](https://www.modelscope.cn/models/Duxiaoman-DI/XuanYuan2-70B/files)
* branch: `master`
* commit id: `45d81a2b09ece9927ed15a35e4ec2983d9cb896e`

将上述url设定的路径下的内容全部下载到`XuanYuan2-70B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of XuanYuan2-70B] \
 --tensor-parallel-size=4 \
 --output-len=20 \
 --max-model-len=16384 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of XuanYuan2-70B] \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --tokenizer=[path of XuanYuan2-70B] \
 --input-len=1024 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16384；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

