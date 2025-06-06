## Auto prefix caching
### 功能介绍
[vLLM auto prefix caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html)

### 离线推理
```shell
python3 -m vllm_utils.offline_inference_with_prefix \
    --model=[path of model] \
    --device=[device type] \
    --max-tokens=128 \
    --tensor-parallel-size=1 \
    --save-output=[file of save inference results]
```

各参数含义如下：
* `--model`：model存储路径；
* `--device`：设备类型，默认为`gcu`；
* `--max-tokens`:推理生成的最多token数量，默认值128，可按需调整；
* `--tensor-parallel-size`:张量并行数，默认值1，可按需调整；
* `--gpu-memory-utilization`：vLLM允许的最大显存占用比例，默认0.9，可按需调整；
* `--save-output`:推理结果的存储文件，默认值为`inference_results_with_prefix.json`,可按需调整，但需保证文件为`json`格式。`default`字段给出未启用`auto prefix caching`时的推理结果，`with prefix`字段给出启用后的推理结果；
* 默认采用graph模式进行推理，可以添加`--enforce-eager`启用eager mode进行推理；

该示例使用内置的`prefix`和`prompts`给出了启用`auto prefix caching`前后的推理结果及耗时.
测试结果表明，在燧原S60 gcu上启用该特性后的推理结果与未启用时的结果一致，且具有一定的加速效果。具体加速比例随模型、`prefix`和`prompts`的不同而变动。

### 性能测试
```shell
python3 -m vllm_utils.benchmark_prefix_caching \
    --model=[path of model] \
    --device=gcu \
    --output-len=128 \
    --block-size=64 \
    --enable-prefix-caching \
    --use-v2-block-manager
```

推理完成后，可以看到两个`cost time`值，两者的比值为开启`auto prefix caching`特性后的加速比。

注：
* `--output-len`可按需调整；

* `--use-v2-block-manager`为可选项；

* 本测试可以看到开启`auto prefix caching`后的正向性能提升，具体提升比例随模型和参数的不同而变动。