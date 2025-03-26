## bloom

### bloom-7b1

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
* branch: `main`
* commit id: `e83e90ba86f87f74aa2731cdab25ccf33976bd66`

- 将上述url设定的路径下的内容全部下载到`bloom-7b1`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of bloom-7b1] \
 --demo=te \
 --output-len=256 \
 --dtype=float16 \
 --max-model-len=2048
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of bloom-7b1] \
 --max-model-len=2048 \
 --tokenizer=[path of bloom-7b1] \
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

### bloomz-7b1

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [bloomz-7b1](https://huggingface.co/bigscience/bloomz-7b1)
* branch: `main`
* commit id: `2f4c4f3ebcf171dbbe2bae989ea2d2f3d3486a97`

- 将上述url设定的路径下的内容全部下载到`bloomz-7b1`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of bloomz-7b1] \
 --demo=te \
 --output-len=20 \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of bloomz-7b1] \
 --max-model-len=2048 \
 --tokenizer=[path of bloomz-7b1] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```

注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### bloomz-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`bloomz-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`bloomz_w8a16_gptq`文件夹中。
- `bloomz_w8a16_gptq`目录结构如下所示：

```shell
bloomz_w8a16_gptq/
      ├── config.json
      ├── model.safetensors
      ├── quantize_config.json
      ├── tokenizer.json
      ├── tokenizer_config.json
      └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of bloomz-w8a16_gptq] \
 --demo=te \
 --output-len=20 \
 --dtype=float16 \
 --quantization=gptq \
 --tensor-parallel-size=8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of bloomz-w8a16_gptq] \
 --tensor-parallel-size=8 \
 --max-model-len=2048 \
 --tokenizer=[path of bloomz-w8a16_gptq] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq
```

注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;