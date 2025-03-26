## Aquila

### Aquila2-34B

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* url: [Aquila2-34B](https://huggingface.co/BAAI/Aquila2-34B/tree/main)
* branch: main
* commit id: 356733caf6221e9dd898cde8ff189a98175526ec

- 将上述url设定的路径下的内容全部下载到`aquila2-34b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of aquila2-34b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --output-len=128 \
 --demo=te \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of aquila2-34b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of aquila2-34b] \
 --input-len=1920 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时，输出内容中的`latency`即为time_to_first_token_latency；

### AquilaChat2-34B

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* url: [AquilaChat2-34B](https://huggingface.co/BAAI/AquilaChat2-34B/tree/main)
* branch: main
* commit id: b9cd9c7436435ab9cfa5e4f009be2b0354979ca8

- 将上述url设定的路径下的内容全部下载到`aquilachat2-34b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of aquilachat2-34b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --output-len=512 \
 --dtype=float16 \
 --demo=ch \
 --template=templates/template_aquilachat2.jinja
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of aquilachat2-34b] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of aquilachat2-34b] \
 --input-len=1024 \
 --output-len=256 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时，输出内容中的`latency`即为time_to_first_token_latency；

### AquilaChat2-34B-16K

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
* url: [AquilaChat2-34B-16K](https://huggingface.co/BAAI/AquilaChat2-34B-16K/tree/main)
* branch: `main`
* commit id: `a06fd164c7170714924d2881c61c8348425ebc94`

- 将上述url设定的路径下的内容全部下载到`aquilachat2-34b-16k`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of aquilachat2-34b-16k] \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --output-len=512 \
 --demo=ch \
 --template=templates/template_aquilachat2.jinja \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of aquilachat2-34b-16k] \
 --tensor-parallel-size=4 \
 --max-model-len=16384 \
 --tokenizer=[path of aquilachat2-34b-16k] \
 --input-len=1024 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为16384；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时，输出内容中的`latency`即为time_to_first_token_latency；
