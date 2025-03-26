## speculative decoding
### 功能介绍
[vLLM speculative decoding](https://docs.vllm.ai/en/latest/models/spec_decode.html)

### 使用方法
```shell
python3 -m vllm_utils.offline_inference_with_speculative_decoding \
    --model=[path of model] \
    --speculative-model=[path of speculative model] \
    --device=[device type] \
    --max-tokens=128 \
    --tensor-parallel-size=1 \
    --num-speculative-tokens=5 \
    --ngram-prompt-lookup-max=3 \
    --demo=[prompts type] \
    --save-output=[file of save inference results]
```

各参数含义如下：
* `--model`：model存储路径；
* `--device`：设备类型，默认为`gcu`；
* `--max-tokens`:推理生成的最多token数量，默认值128，可按需调整；
* `--tensor-parallel-size`:张量并行数，默认值1，可按需调整；
* `--num-speculative-tokens`:speculative产生的tokens的数量，默认值5，可按需调整；
* `--ngram-prompt-lookup-max`:`ngram`模式下产生的最多token数量，默认值为3，可按需调整；
* `--template`:chat类模型使用的模板文件，若不设置则使用默认模板；
* `--gpu-memory-utilization`：vLLM允许的最大显存占用比例，默认0.9，可按需调整；
* `--demo`:使用的示例模型，可用值有`te`、`tc`、`chat`等，分别对应内置的英文、中文和对话等prompts；
* `--save-output`:推理结果的存储文件，默认值为`inference_results_with_speculative.json`,可按需调整，但需保证文件为`json`格式。生成的json中，`default`字段给出默认的推理结果，`speculative`字段给出启用`speculative decoding`后的推理结果，`ngram`字段给出启用`ngram`模式时的推理结果；
* 添加`--trust-remote-code`启用该特性；
* 默认采用graph模式进行推理，可以添加`--enforce-eager`启用eager mode进行推理；

该示例使用内置的`prompts`给出了启用`speculative decoding`与`ngram`后相比未启用时的推理结果及耗时。
测试结果表明，在燧原S60 gcu上启用该特性后的推理结果与未启用时的结果一致，且具有一定的加速效果。具体加速比例随模型和`prompts`的不同而变动。

### 性能测试
#### Speculating with a draft model
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
    --model=[path of model] \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --device gcu \
    --block-size 64 \
    --speculative-model [path of draft model] \
    --use-v2-block-manager \
    --num-speculative-tokens 5

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
* 服务端参数`--dtype`、`--num-speculative-tokens`可按需调整；

* 客户端参数`--num-prompts`、`--random-input-len`、`--random-output-len`、`--request-rate`可按需调整；

* 需根据模型推理性能，调整上述参数，方可取得正向性能提升；

#### Speculating by matching ngrams in the prompt
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
    --model=[path of model] \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --device gcu \
    --block-size 64 \
    --speculative-model [ngram] \
    --use-v2-block-manager \
    --num-speculative-tokens 5 \
    --ngram-prompt-lookup-max 3

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
* 服务端参数`--dtype`、`--num-speculative-tokens`、`--ngram-prompt-lookup-max`可按需调整；

* 客户端参数`--num-prompts`、`--random-input-len`、`--random-output-len`、`--request-rate`可按需调整；

* 需根据模型推理性能，调整上述参数，方可取得正向性能提升；