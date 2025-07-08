## Chunked prefill
### 功能介绍
[vLLM chunked prefill](https://docs.vllm.ai/en/latest/models/performance.html#chunked-prefill)

### 使用方法
```shell
python3 -m vllm_utils.offline_inference_with_chunked_prefill \
    --model=[path of model] \
    --device=[device type] \
    --max-tokens=128 \
    --tensor-parallel-size=1
```

各参数含义如下：
* `--model`：model存储路径；
* `--device`：设备类型，默认为`gcu`；
* `--max-tokens`:推理生成的最多token数量，默认值128，可按需调整；
* `--max-num-batched-tokens`:推理时，一个batch中最多的token数量，默认值256，可按需向上调整；
* `--tensor-parallel-size`:张量并行数，默认值1，可按需调整；
* `--gpu-memory-utilization`：vLLM允许的最大显存占用比例，默认0.9，可按需调整；
* 默认采用graph模式进行推理，可以添加`--enforce-eager`启用eager mode进行推理；

该示例使用内置的`prefix`和`prompts`给出了启用`chunked prefill`功能时的推理效果。

### 性能测试
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
    --model=[path of model] \
    --dtype float16 \
    --max-model-len=8192 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --device gcu \
    --block-size 64 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 1024

# 启动客户端
python3 -m vllm_utils.benchmark_serving \
    --backend vllm \
    --dataset-name random \
    --model=[path of model] \
    --num-prompts 10 \
    --random-input-len 2048 \
    --random-output-len 300 \
    --request-rate 0.1 \
    --trust-remote-code
```

注：
* 服务端参数`--dtype`、`--tensor-parallel-size`、`--max-num-batched-tokens`可按需调整，若不设置，则`--max-num-batched-tokens`默认值为512；

* 客户端参数`--num-prompts`、`--random-input-len`、`--random-output-len`、`--request-rate`可按需调整，但需保证`--random-input-len`值大于`--max-num-batched-tokens`；

* 需结合模型推理耗时，调整上述参数，确保`chunked prefill` feature使能并产生正向性能提升；