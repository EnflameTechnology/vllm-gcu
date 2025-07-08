## Mistral

### Mixtral-8x22B-Instruct-v0.1

本模型推理及性能测试需要8张enflame gcu。
#### 模型下载
* url:[Mixtral-8x22B-Instruct-v0.1](https://www.modelscope.cn/models/AI-ModelScope/Mixtral-8x22B-Instruct-v0.1/files)
* branch: `master`
* commit id: `eb269184`

将上述url设定的路径下的内容全部下载到`Mixtral-8x22B-Instruct-v0.1`文件夹中。

#### 安装依赖
```shell
python3 -m pip install transformers==4.48.2
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Mixtral-8x22B-Instruct-v0.1] \
 --tensor-parallel-size=8 \
 --output-len=128 \
 --demo=te \
 --dtype=bfloat16 \
 --device=gcu \
 --max-model-len=32768 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Mixtral-8x22B-Instruct-v0.1] \
 --device=gcu \
 --max-model-len=32768 \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=bfloat16 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization 0.945 \
 --trust-remote-code
```
注：
*  本模型支持的`max-model-len`为65536；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;