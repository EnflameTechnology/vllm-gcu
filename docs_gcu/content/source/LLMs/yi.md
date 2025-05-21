## Yi

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

### Yi-34B-Chat-w8a16_gptq

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Yi-34B-Chat-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Yi-34B-Chat-w8a16_gptq`文件夹中。
- `Yi-34B-Chat-w8a16_gptq`目录结构如下所示：

```shell
Yi-34B-Chat-w8a16_gptq/
            ├── config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── tokenizer.json
            ├── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Yi-34B-Chat-w8a16_gptq] \
 --tensor-parallel-size=2 \
 --output-len=256 \
 --demo=te \
 --dtype=float16
```

#### serving模式

注：需要安装以下依赖：

```shell
python3 -m pip install numba==0.60.0
```

```shell

# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of Yi-34B-Chat-w8a16_gptq] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --disable-log-requests \
 --gpu-memory-utilization=0.9 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq \
 --tokenizer-mode slow

# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend=vllm \
 --dataset-name=random \
 --model=[path of Yi-34B-Chat-w8a16_gptq] \
 --num-prompts=10 \
 --random-input-len=4 \
 --random-output-len=300 \
 --trust-remote-code

```
注：
*  为保证输入输出长度固定，数据集使用随机数测试；

*  num-prompts, random-input-len和random-output-len可按需调整；

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
