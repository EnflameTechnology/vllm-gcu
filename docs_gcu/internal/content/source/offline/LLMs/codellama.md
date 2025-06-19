## codellama



### codellama-13b-hf

#### 模型下载
*  url: [codellama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf/tree/main)

*  branch: `main`

*  commit id: `9d8db7d`

将上述url设定的路径下的内容全部下载到`codellama-13b-hf`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-13b-hf] \
 --demo=cc \
 --dtype=float16 \
 --output-len=256 \
 --max-model-len=1024
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-13b-hf] \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-13b-hf] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下单卡支持的`max-model-len`为16k，ecc on模式下单卡支持的`max-model-len`为10k，使能16k需要双卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-13b-python

#### 模型下载
*  url: [codellama-13b-python](https://huggingface.co/codellama/CodeLlama-13b-Python-hf/tree/main)

*  branch: `main`

*  commit id: `832ed72`

将上述url设定的路径下的内容全部下载到`codellama-13b-python`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-13b-python] \
 --demo=cc \
 --dtype=float16 \
 --output-len=256 \
 --max-model-len=1024
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-13b-python] \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-13b-python] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下单卡支持的`max-model-len`为16k，ecc on模式下单卡支持的`max-model-len`为10k，使能16k需要双卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-13b-instruct

#### 模型下载
*  url: [codellama-13b-instruct](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf/tree/main)

*  branch: `main`

*  commit id: `daacef3`

将上述url设定的路径下的内容全部下载到`codellama-13b-instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-13b-instruct] \
 --demo=cin \
 --template=templates/template_default.jinja \
 --dtype=float16 \
 --max-model-len=1024 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-13b-instruct] \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-13b-instruct] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下单卡支持的`max-model-len`为16k，ecc on模式下单卡支持的`max-model-len`为10k，使能16k需要双卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-34b-hf

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [codellama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf/tree/main)

*  branch: `main`

*  commit id: `8212871`

将上述url设定的路径下的内容全部下载到`codellama-34b-hf`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-34b-hf] \
 --tensor-parallel-size=2 \
 --demo=cin \
 --template=default \
 --dtype=float16 \
 --output-len=256
``` 

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-34b-hf] \
 --tensor-parallel-size=2 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-34b-hf] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下双卡支持的`max-model-len`为16k，ecc on模式下双卡支持的`max-model-len`为10k，使能16k需要四卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-34b-python

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [codellama-34b-python](https://huggingface.co/codellama/CodeLlama-34b-Python-hf/tree/main)

*  branch: `main`

*  commit id: `0d7350d`

将上述url设定的路径下的内容全部下载到`codellama-34b-python`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-34b-python] \
 --tensor-parallel-size=2 \
 --demo=cin \
 --template=templates/template_default.jinja \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-34b-python] \
 --tensor-parallel-size=2 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-34b-python] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下双卡支持的`max-model-len`为16k，ecc on模式下双卡支持的`max-model-len`为10k，使能16k需要四卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-34b-instruct

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [codellama-34b-instruct](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf/tree/main)

*  branch: `main`

*  commit id: `bf5e506`

将上述url设定的路径下的内容全部下载到`codellama-34b-instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-34b-instruct] \
 --tensor-parallel-size=2 \
 --demo=cin \
 --template=default \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-34b-instruct] \
 --tensor-parallel-size=2 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-34b-instruct] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型在ecc off模式下双卡支持的`max-model-len`为16k，ecc on模式下双卡支持的`max-model-len`为10k，使能16k需要四卡；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-70b-hf

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
*  url: [codellama-70b-hf](https://huggingface.co/codellama/CodeLlama-70b-hf/tree/main)

*  branch: `main`

*  commit id: `4570a4e`

将上述url设定的路径下的内容全部下载到`codellama-70b-hf`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-70b-hf] \
 --tensor-parallel-size=4 \
 --demo=cc \
 --output-len=256 \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-70b-hf] \
 --tensor-parallel-size=4 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-70b-hf] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-70b-python

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
*  url: [codellama-70b-python](https://huggingface.co/codellama/CodeLlama-70b-Python-hf/tree/main)

*  branch: `main`

*  commit id: `7946798`

将上述url设定的路径下的内容全部下载到`codellama-70b-python`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-70b-python] \
 --tensor-parallel-size=4 \
 --demo=cc \
 --output-len=256 \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-70b-python] \
 --tensor-parallel-size=4 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-70b-python] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-70b-instruct

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
*  url: [codellama-70b-instruct](https://huggingface.co/codellama/CodeLlama-70b-Instruct-hf/tree/main)

*  branch: `main`

*  commit id: `b256b38`

将上述url设定的路径下的内容全部下载到`codellama-70b-instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama-70b-instruct] \
 --tensor-parallel-size=4 \
 --demo=cc \
 --output-len=256 \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama-70b-instruct] \
 --tensor-parallel-size=4 \
 --max-model-len=1024 \
 --tokenizer=[path of codellama-70b-instruct] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### codellama-70b-instruct-w8a16_gptq

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`codellama-70b-instruct-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`codellama_70b_instruct_w8a16_gptq`文件夹中。
- `codellama_70b_instruct_w8a16_gptq`目录结构如下所示：

```shell
codellama_70b_instruct_w8a16_gptq/
            ├── added_tokens.json
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── special_tokens_map.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── tokenizer.model
            └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama_70b_instruct_w8a16_gptq] \
 --tensor-parallel-size=2 \
 --demo=cc \
 --output-len=256 \
 --dtype=float16  \
 --quantization gptq
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama_70b_instruct_w8a16_gptq] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of codellama_70b_instruct_w8a16_gptq] \
 --input-len=1024 \
 --output-len=3072 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
