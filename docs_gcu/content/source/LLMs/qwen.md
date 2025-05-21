## Qwen

### Qwen1.5-7B
#### 模型下载
*  url: [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B/tree/main)

*  branch: `main`

*  commit id: `e52fa2e`

将上述url设定的路径下的内容全部下载到`Qwen1.5-7B`文件夹中。

#### 安装依赖
```shell
python3 -m pip install tiktoken
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of hf_Qwen1.5-7B_model] \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=16384
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of hf_Qwen1.5-7B_model] \
 --max-model-len=16384 \
 --tokenizer=[path of hf_Qwen1.5-7B_model] \
 --input-len=8192 \
 --output-len=8192 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16384；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url: [Qwen1.5-32B](https://huggingface.co/Qwen/Qwen1.5-32B/tree/main)

*  branch: `main`

*  commit id: `cefef80dc06a65f89d1d71d0adbc56d335ca2490`

将上述url设定的路径下的内容全部下载到`Qwen1.5-32B`文件夹中。

#### 安装依赖
```shell
python3 -m pip install tiktoken
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of hf_Qwen1.5-32B_model] \
 --tensor-parallel-size=2 \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=2048
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of hf_Qwen1.5-32B_model] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of hf_Qwen1.5-32B_model] \
 --input-len=2048 \
 --output-len=2048 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-72B-Chat

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [Qwen1.5-72B-Chat](https://huggingface.co/Qwen/Qwen1.5-72B-Chat/tree/main)

*  branch: `main`

*  commit id: `1a6ccc1215278f962c794b1848c710c29ef4053d`

将上述url设定的路径下的内容全部下载到`Qwen1.5-72B-Chat`文件夹中。

#### 安装依赖
```shell
python3 -m pip install tiktoken
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of hf_Qwen1.5-72B-Chat_model] \
 --tensor-parallel-size=4 \
 --output-len=20 \
 --demo=te \
 --dtype=float16 \
 --max-model-len=2048 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of hf_Qwen1.5-72B-Chat_model] \
 --tensor-parallel-size=4 \
 --max-model-len=2048 \
 --tokenizer=[path of hf_Qwen1.5-72B-Chat_model] \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为2048；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-14B-Chat-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`QWen1.5-14b-chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`QWen1.5-14b-chat_w8a16_gptq`文件夹中。
- `QWen1.5-14b-chat_w8a16_gptq`目录结构如下所示：

```shell
QWen1.5-14b-chat_w8a16_gptq/
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── merges.txt
            ├── tops_quantize_info.json
            └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of QWen1.5-14b-chat_w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=20 \
 --max-model-len=2048
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of QWen1.5-14b-chat_w8a16_gptq] \
 --input-len=4096 \
 --output-len=4096 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=8192 \
 --dtype=float16 \
 --quantization=gptq \
 --enforce-eager
```

注:
* 单张gcu上可以支持的`max-model-len`为16k，若需使用到模型自身支持的32k的`max-model-len`，则需设置`--tensor-parallel-size=2`；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`qwen1.5-32b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`qwen1.5_32b_w8a16_gptq`文件夹中。
- `qwen1.5_32b_w8a16_gptq`目录结构如下所示：

```shell
qwen1.5_32b_w8a16_gptq/
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── merges.txt
            ├── vocab.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of qwen_1_5_32b_w8a16_gptq] \
 --demo=te \
 --output-len=256 \
 --dtype=float16  \
 --quantization gptq \
 --max-model-len 4096
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of qwen_1_5_32b_w8a16_gptq] \
 --max-model-len=4096 \
 --tokenizer=[path of qwen_1_5_32b_w8a16_gptq] \
 --input-len=2048 \
 --output-len=2048 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

#### serving模式
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of qwen1.5_32b_w8a16_gptq]  \
 --tensor-parallel-size=1 \
 --max-model-len=4096  \
 --disable-log-requests  \
 --gpu-memory-utilization=0.9  \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq

# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend=vllm  \
 --dataset-name=random  \
 --model=[path of qwen1.5_32b_w8a16_gptq]  \
 --num-prompts=10  \
 --random-input-len=4   \
 --random-output-len=300  \
 --trust-remote-code
```
注：
* 为保证输入输出长度固定，数据集使用随机数测试；
* num-prompts, random-input-len和random-output-len可按需调整；

### Qwen1.5-MoE-A2.7B

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/tree/main)

*  branch: `main`

*  commit id: `1a758c5`

将上述url设定的路径下的内容全部下载到`Qwen1.5-MoE-A2.7B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-MoE-A2.7B] \
 --output-len=256 \
 --demo=te \
 --dtype=bfloat16 \
 --max-model-len=8192
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-MoE-A2.7B] \
 --max-model-len=8192 \
 --tokenizer=[path of Qwen1.5-MoE-A2.7B] \
 --input-len=4096 \
 --output-len=4096 \
 --num-prompts=2 \
 --block-size=64 \
 --dtype=bfloat16
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Qwen2-7B

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen2-7B](https://www.modelscope.cn/models/qwen/Qwen2-7B)

*  branch: `main`

*  commit id: `da7ff8fb`

将上述url设定的路径下的内容全部下载到`Qwen2-7B`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Qwen2-7B] \
    --tokenizer=[path of Qwen2-7B] \
    --num-prompts 1 \
    --max-model-len=32768 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=1 \
    --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-7B] \
    --tensor-parallel-size 1 \
    --max-model-len=32768 \
    --input-len=31000 \
    --output-len=1000 \
    --dtype=float16 \
    --device gcu \
    --num-prompts=4 \
    --block-size=64 \
    --gpu-memory-utilization=0.9
```

注:
* 单张gcu上可以支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen-7B-Instruct

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen2-7B-Instruct](https://www.modelscope.cn/models/qwen/Qwen2-7B-Instruct/)

*  branch: `main`

*  commit id: `39c0a5ab`

将上述url设定的路径下的内容全部下载到`Qwen2-7B-Instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Qwen2-7B-Instruct] \
    --tokenizer=[path of Qwen2-7B-Instruct] \
    --num-prompts 1 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=1 \
    --max-model-len=32768 \
    --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-7B-Instruct] \
    --tensor-parallel-size 1 \
    --max-model-len=32768 \
    --input-len=8000 \
    --output-len=8000 \
    --dtype=bfloat16 \
    --device gcu \
    --num-prompts=1 \
    --block-size=64 \
    --gpu-memory-utilization=0.9
```

注:
* 单张gcu上可以支持的`max-model-len`为32768；
* `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2-72B-padded-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen2-72B-padded-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen2-72B-padded-w8a16_gptq`文件夹中。
- `Qwen2-72B-padded-w8a16_gptq`目录结构如下所示：

```shell
Qwen2-72B-padded-w8a16_gptq
├── config.json
├── generation_config.json
├── merges.txt
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
├── tops_quantize_info.json
└── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Qwen2-72B-padded-w8a16_gptq] \
    --tokenizer=[path of Qwen2-72B-padded-w8a16_gptq] \
    --num-prompts 1 \
    --max-model-len=32768 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.9
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-72B-padded-w8a16_gptq]\
    --tensor-parallel-size 8 \
    --max-model-len=32768 \
    --input-len=8000 \
    --output-len=8000  \
    --dtype=float16 \
    --device gcu \
    --num-prompts=1  \
    --block-size=64 \
    --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Qwen2-72B-Instruct

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [Qwen2-72B-Instruct](https://www.modelscope.cn/models/qwen/Qwen2-72B-Instruct/)

*  branch: `main`

*  commit id: `da7ff8fb`

将上述url设定的路径下的内容全部下载到`Qwen2-72B-Instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Qwen2-72B-Instruct]  \
    --tokenizer=[path of Qwen2-72B-Instruct]  \
    --num-prompts 1 \
    --block-size=64 \
    --max-model-len=32768 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --tensor-parallel-size=8 \
    --gpu-memory-utilization=0.9
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model [path of Qwen2-72B-Instruct] \
    --tensor-parallel-size 8 \
    --max-model-len=32768 \
    --input-len=8000 \
    --output-len=8000 \
    --dtype=float16 \
    --device gcu \
    --num-prompts=1 \
    --block-size=64 \
    --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2-1.5B-Instruct

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct/tree/main/)

*  branch: `main`

*  commit id: `ba1cf18`

将上述url设定的路径下的内容全部下载到`Qwen2-1.5B-Instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2-1.5B-Instruct] \
 --tensor-parallel-size=1 \
 --max-model-len=32768 \
 --dtype=float16 \
 --device=gcu \
 --output-len=512 \
 --demo=te \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen2-1.5B-Instruct] \
 --tokenizer=[path of Qwen2-1.5B-Instruct] \
 --tensor-parallel-size=1 \
 --max-model-len=32768 \
 --dtype=float16 \
 --device=gcu \
 --input-len=2048 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.9
```

注:
* 单张gcu上可以支持的`max-model-len`为32768；
* `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-4B-Chat

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen1.5-4B-Chat](https://huggingface.co/Qwen/Qwen1.5-4B-Chat)

*  branch: `main`

*  commit id: `a7a4d4945d28bac955554c9abd2f74a71ebbf22f`

将上述url设定的路径下的内容全部下载到`Qwen1.5-4B-Chat`文件夹中。

#### 安装依赖
```shell
python3 -m pip install tiktoken
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-4B-Chat_model] \
 --max-model-len=32768 \
 --tensor-parallel-size=1 \
 --output-len=256 \
 --demo=te \
 --dtype=bfloat16 \
 --num-prompts=1
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-4B-Chat_model] \
 --tokenizer=[path of Qwen1.5-4B-Chat_model] \
 --input-len=2048 \
 --output-len=1024 \
 --tensor-parallel-size=1 \
 --max-model-len=32768 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;



### Qwen1.5-32B-Chat-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-32B-Chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen1.5-32B-Chat-w8a16_gptq`文件夹中。
- `Qwen1.5-32B-Chat-w8a16_gptq`目录结构如下所示：

```shell
Qwen1.5-32B-Chat-w8a16_gptq
  ├── config.json
  ├── model.safetensors
  ├── quantize_config.json
  ├── tokenizer_config.json
  ├── tokenizer.json
  ├── tops_quantize_info.json
  └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-32B-Chat-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --tensor-parallel-size=2 \
 --num-prompts=1 \
 --gpu-memory-utilization=0.9 \
 --block-size=64
```

#### serving模式
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server --model=[path of Qwen1.5-32B-Chat-w8a16_gptq]  \
 --tensor-parallel-size 2 \
 --max-model-len=4096  \
 --disable-log-requests  \
 --gpu-memory-utilization=0.9  \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq

# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend=vllm  \
 --dataset-name=random  \
 --model=[path of Qwen1.5-32B-Chat-w8a16_gptq]  \
 --num-prompts=10  \
 --random-input-len=4   \
 --random-output-len=300  \
 --trust-remote-code
```
注：
* 为保证输入输出长度固定，数据集使用随机数测试；
* num-prompts, random-input-len和random-output-len可按需调整；


### Qwen1.5-72B-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-72B-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen1.5-72B-w8a16_gptq`文件夹中。
- `Qwen1.5-72B-w8a16_gptq`目录结构如下所示：

```shell
Qwen1.5-72B-w8a16_gptq
  ├── config.json
  ├── model.safetensors
  ├── quantize_config.json
  ├── tokenizer_config.json
  ├── tokenizer.json
  ├── tops_quantize_info.json
  └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-72B-w8a16_gptq_model] \
 --demo=te \
 --dtype=float16 \
 --tensor-parallel-size=8 \
 --output-len=256 \
 --block-size=16 \
 --num-prompts=1 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-72B-w8a16_gptq_model] \
 --input-len=1024 \
 --output-len=3027 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --tensor-parallel-size=8 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```

注:
* 八张gcu上可以支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Qwen1.5-72B-Chat-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-72B-Chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen1.5-72B-Chat-w8a16_gptq`文件夹中。
- `Qwen1.5-72B-Chat-w8a16_gptq`目录结构如下所示：

```shell
Qwen1.5-72B-Chat-w8a16_gptq
  ├── config.json
  ├── generation_config.json
  ├── merges.txt
  ├── model.safetensors
  ├── quantize_config.json
  ├── tokenizer_config.json
  ├── tokenizer.json
  ├── tops_quantize_info.json
  └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-72B-Chat-w8a16_gptq_model] \
 --demo=te \
 --dtype=float16 \
 --tensor-parallel-size=8 \
 --output-len=256 \
 --num-prompts=1 \
 --block-size=64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-72B-Chat-w8a16_gptq_model] \
 --input-len=1024 \
 --output-len=3027 \
 --num-prompts=1 \
 --block-size=64 \
 --tensor-parallel-size=8 \
 --max-model-len=32768 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```

注:
* 八卡gcu上可以支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B-w4a16

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-32B-w4a16.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen1.5_32B_w4a16_gptq`文件夹中。
- `Qwen1.5_32B_w4a16_gptq`目录结构如下所示：

```shell
Qwen1.5_32B_w4a16_gptq/
  ├── config.json
  ├── vocab.json
  ├── generation_config.json
  ├── model.safetensors
  ├── quantize_config.json
  ├── merges.txt
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5_32B_w4a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5_32B_w4a16_gptq] \
 --input-len=1024 \
 --output-len=500 \
 --num-prompts=1 \
 --max-model-len=32768 \
 --dtype=float16 \
 --quantization=gptq
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### qwen2-72b-instruct-gptq-int4

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [qwen2-72b-instruct-gptq-int4](https://modelscope.cn/models/qwen/qwen2-72b-instruct-gptq-int4)

*  branch: `master`

*  commit id: `c7e75f6b`

将上述url设定的路径下的内容全部下载到`qwen2-72b-instruct-gptq-int4`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of qwen2-72b-instruct-gptq-int4] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --output-len=512 \
 --demo=te \
 --dtype=float16 \
 --device=gcu \
 --quantization=gptq
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of qwen2-72b-instruct-gptq-int4] \
 --max-model-len=32768 \
 --tokenizer=[path of qwen2-72b-instruct-gptq-int4] \
 --input-len=1024 \
 --output-len=500 \
 --num-prompts=1 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --quantization=gptq \
 --device=gcu
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### qwen2-72b-instruct-gptq-int8

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [qwen2-72b-instruct-gptq-int8](https://www.modelscope.cn/models/Qwen/Qwen2-72B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `6fcdadb8`

将上述url设定的路径下的内容全部下载到`qwen2-72b-instruct-gptq-int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of qwen2-72b-instruct-gptq-int8] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device=gcu \
 --gpu-memory-utilization 0.9 \
 --quantization=gptq \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of qwen2-72b-instruct-gptq-int8] \
 --max-model-len=32768 \
 --dtype=float16 \
 --gpu-memory-utilization 0.9 \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --tensor-parallel-size=4 \
 --block-size=64 \
 --quantization=gptq \
 --device=gcu \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B-w4a16c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-32B-w4a16c8.tar`文件以及并解压，将压缩包内的内容全部拷贝到`Qwen1.5_32B_w4a16c8`文件夹中。
- `Qwen1.5_32B_w4a16c8`目录结构如下所示：

```shell
Qwen1.5_32B_w4a16c8/
  ├── config.json
  ├── vocab.json
  ├── generation_config.json
  ├── model.safetensors
  ├── quantize_config.json
  ├── int8_kv_cache.json
  ├── merges.txt
  ├── tokenizer.json
  ├── tokenizer_config.json
  └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5_32B_w4a16c8] \
 --demo=te \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5_32B_w4a16c8] \
 --input-len=1024 \
 --output-len=500 \
 --num-prompts=1 \
 --max-model-len=32768 \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2-72B-Instruct-w4a16c8

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url: [Qwen2-72B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-72B-Instruct-GPTQ-Int4/tree/main)

*  branch: `main`

*  commit id: `9d456ed`

* 另外需要下载int8_kv_cache.json, 联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载
- 将下载的`Qwen2-72B-Instruct-GPTQ-Int4`和`int8_kv_cache.json`放入`Qwen2_72B_Instruct_w4a16c8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2_72B_Instruct_w4a16c8] \
 --demo=te \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8 \
 --output-len=256 \
 --tensor-parallel-size=2 \
 --max-model-len=32768
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen2_72B_Instruct_w4a16c8] \
 --input-len=1024 \
 --output-len=500 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8 \
 --tensor-parallel-size=2 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### qwen1.5-72b-chat-awq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [qwen1.5-72b-chat-awq](https://modelscope.cn/models/qwen/qwen1.5-72b-chat-awq/files)

*  branch: `master`

*  commit id: `4b52e410`


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of qwen1.5-72b-chat-awq] \
 --demo=te \
 --dtype=float16 \
 --tensor-parallel-size 8 \
 --output-len=128 \
 --device gcu \
 --quantization=awq \
 --gpu-memory-utilization 0.9 \
 --max-model-len=32768 \
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of qwen1.5-72b-chat-awq] \
 --tensor-parallel-size 8 \
 --input-len=20480 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --dtype=float16 \
 --device gcu \
 --quantization=awq \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2-57B-A14B

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
*  url: [Qwen2-57B-A14B](https://modelscope.cn/models/qwen/Qwen2-57B-A14B/files)

*  branch: `master`

*  commit id: `d8cb5700`


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2-57B-A14B] \
 --demo=te \
 --dtype=bfloat16 \
 --tensor-parallel-size 4 \
 --max-model-len=8192 \
 --output-len=512 \
 --device gcu \
 --gpu-memory-utilization=0.9
```

### Qwen1.5-110B-Chat-w8a16_gptq

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`QWen1.5-110B-Chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`QWen1.5-110B-Chat_w8a16_gptq`文件夹中。
- `QWen1.5-110B-Chat_w8a16_gptq`目录结构如下所示：

```shell
QWen1.5-110B-Chat_w8a16_gptq/
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── merges.txt
            ├── tops_quantize_info.json
            └── vocab.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5-110B-Chat] \
 --tensor-parallel-size=8 \
 --output-len=256 \
 --demo=te \
 --dtype=float16 \
 --block-size=64 \
 --num-prompts=1 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-110B-Chat] \
 --max-model-len=32768 \
 --tokenizer=[path of Qwen1.5-110B-Chat] \
 --input-len=1000 \
 --output-len=3000 \
 --tensor-parallel-size=8 \
 --num-prompts=1 \
 --block-size=16 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B-Chat-w4a16c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Qwen1.5-32B-Chat-GPTQ-Int4](https://www.modelscope.cn/models/qwen/qwen1.5-32b-chat-gptq-int4 )

*  branch: `main`

*  commit id: `226cd6ec86d885563fb5c7c2c4560a035564f20f`

* 另外需要下载int8_kv_cache.json, 联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载
- 将下载的`Qwen1.5-32B-Chat-GPTQ-Int4`和`int8_kv_cache.json`放入`Qwen1.5_32B_Chat_w4a16c8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen1.5_32B_Chat_w4a16c8] \
 --demo=te \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8 \
 --output-len=256 \
 --max-model-len=32768
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5_32B_Chat_w4a16c8] \
 --input-len=1024 \
 --output-len=32 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=32768 \
 --dtype=float16 \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --kv-cache-dtype=int8 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2-72B-w8a8c8

本模型推理及性能测试需要4张enflame gcu。

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen2-72B-w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen2-72B-w8a8c8`文件夹中。
- `Qwen2-72B-w8a8c8`目录结构如下所示：

```shell
Qwen2-72B-w8a8c8/

```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2-72B-w8a8c8] \
 --max-model-len=32768 \
 --demo=te \
 --dtype=bfloat16 \
 --tensor-parallel-size=4 \
 --quantization-param-path=[path of Qwen2-72B-w8a8c8] \
 --kv-cache-dtype=int8 \
 --output-len=256
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of Qwen2-72B-w8a8c8]  \
 --max-model-len=32768  \
 --tensor-parallel-size=4 \
 --disable-log-requests  \
 --gpu-memory-utilization=0.9  \
 --block-size=64 \
 --dtype=bfloat16 \
 --kv-cache-dtype=int8 \
 --quantization-param-path=[path of Qwen2-72B-w8a8c8]


# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend=vllm  \
 --dataset-name=random  \
 --model=[path of Qwen2-72B-w8a8c8]  \
 --num-prompts=1  \
 --random-input-len=3000 \
 --random-output-len=1000 \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen1.5-32B-w8a8c8

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Qwen1.5-32B-w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`Qwen1.5-32B-w8a8`文件夹中。
- int8_kv_cache.json文件请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载，并拷贝到`Qwen1.5-32B-w8a8`文件夹中。
- `Qwen1.5-32B-w8a8`目录结构如下所示：

```shell
Qwen1.5-32B-w8a8
├── config.json
├── generation_config.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
├── tops_quantize_info.json
└── vocab.json
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --demo=te \
 --model=[path of Qwen1.5-32B-w8a8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --kv-cache-dtype int8 \
 --quantization=w8a8 \
 --max-model-len=32768 \
 --tensor-parallel-size 2 \
 --dtype=float16 \
 --output-len=128 \
 --device gcu \
 --gpu-memory-utilization 0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Qwen1.5-32B-w8a8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --kv-cache-dtype int8 \
 --tensor-parallel-size 2 \
 --max-model-len=32768 \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-32B-Instruct-GPTQ-Int8_w8a16

#### 模型下载
*  url: [Qwen2.5-32B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `996af7d8`

将上述url设定的路径下的内容全部下载到`Qwen2.5-32B-Instruct-GPTQ-Int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size 2 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-72B-Instruct-GPTQ-Int8_w8a16

#### 模型下载
*  url: [Qwen2.5-72B-Instruct-GPTQ-Int8](https://modelscope.cn/models/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `ce39716d`

将上述url设定的路径下的内容全部下载到`Qwen2.5-72B-Instruct-GPTQ-Int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --tensor-parallel-size 4 \
 --max-seq-len-to-capture 32768 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-72B-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-0.5B-Instruct

#### 模型下载
*  url: [Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files)

*  branch: `master`

*  commit id: `42b20c45`

将上述url设定的路径下的内容全部下载到`Qwen2.5-0.5B-Instruct`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-0.5B-Instruct] \
 --tensor-parallel-size 1 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen2.5-0.5B-Instruct] \
 --tensor-parallel-size 1 \
 --max-seq-len-to-capture=32768 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-0.5B-Instruct] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen2.5-Coder-32B-Instruct
#### 模型下载
*  url: [Qwen2.5-Coder-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-Coder-32B-Instruct/files)

*  branch: `master`

*  commit id: `19b16075`

将上述url设定的路径下的内容全部下载到`Qwen2.5-Coder-32B-Instruct`文件夹中。

#### 批量离线推理
```shell
 python3.10 -m vllm_utils.benchmark_test \
 --model=[path of Qwen2.5-coder-32b-instruct] \
 --demo=te \
 --tensor-parallel-size 8 \
 --max-model-len=32768 \
 --output-len=128 \
 --dtype=bfloat16 \
 --device gcu \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code
```

#### 性能测试
```shell
 python3.10 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of Qwen2.5-coder-32b-instruct] \
 --tensor-parallel-size 8 \
 --max-model-len=32768 \
 --input-len=9216 \
 --output-len=9216 \
 --dtype=bfloat16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.9 \
 --trust-remote-code

```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### QwQ-32B
#### 模型下载
*  url: [QwQ-32B](https://modelscope.cn/models/Qwen/QwQ-32B/files)

*  branch: `master`

*  commit id: `887ddbd72be5bed61cd702b87cd2fc25e65a708d`

将上述url设定的路径下的内容全部下载到`QwQ-32B`文件夹中。

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
 --trust-remote-code
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
  --dtype=bfloat16


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
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### QWen3-30B-A3B

#### 模型下载
*  url: [QWen3-30B-A3B](https://www.modelscope.cn/models/Qwen/QWen3-30B-A3B/files)

*  branch: `master`

*  commit id: `e34b3e98`

将上述url设定的路径下的内容全部下载到`QWen3-30B-A3B`文件夹中。

#### 批量离线推理
```shell
  python3 -m vllm_utils.benchmark_test \
  --model [path of QWen3-30B-A3B] \
  --tensor-parallel-size=2 \
  --max-model-len=32768 \
  --output-len=128 \
  --demo=te \
  --dtype=bfloat16 \
  --device gcu
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
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'


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
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### QWen3-30B-A3B-AWQ

#### 模型下载
*  url: [QWen3-30B-A3B-AWQ](https://modelscope.cn/models/swift/Qwen3-30B-A3B-AWQ/files)

*  branch: `master`

*  commit id: `3441b6ac`

将上述url设定的路径下的内容全部下载到`QWen3-30B-A3B-AWQ`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of QWen3-30B-A3B-AWQ] \
 --tensor-parallel-size 2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code \
 --quantization=moe_wna16_gcu
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of QWen3-30B-A3B-AWQ] \
 --tensor-parallel-size 2 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefil \
 --quantization=moe_wna16_gcu


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of QWen3-30B-A3B-AWQ] \
 --num-prompts 1 \
 --random-input-len 130048 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-token
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Qwen3-32B-AWQ

#### 模型下载
*  url: [Qwen3-32B-AWQ](https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ)

*  branch: `master`

*  commit id: `f8fa721f`

将上述url设定的路径下的内容全部下载到`Qwen3-32B-AWQ`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model [path of Qwen3-32B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len=40960 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device gcu \
 --trust-remote-code
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [path of Qwen3-32B-AWQ] \
 --tensor-parallel-size 4 \
 --max-model-len 131072 \
 --disable-log-requests \
 --gpu-memory-utilization 0.9 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --enable-chunked-prefil


# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen3-32B-AWQ] \
 --num-prompts 1 \
 --random-input-len 130048 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-token
```
注：
*  本模型支持的`max-model-len`为131072；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;