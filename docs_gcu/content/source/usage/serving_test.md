## serving mode
vLLM-gcu的serving mode测试可以参考[vLLM官方手册](https://docs.vllm.ai/en/v0.8.0/)的Inference and Serving章节内容。另外，vLLM-gcu对vLLM官方的[benchmark_serving](https://github.com/vllm-project/vllm/blob/v0.8.0/benchmarks/benchmark_serving.py)测试做了一些修改，以满足不同场景下的serving mode性能测试。


### 修改内容

- 开放更多sampler相关测试参数：`top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `ignore_eos`, `include_stop_str_in_output`, `keep_special_tokens`
- 新增控制input/output长度参数：`strict_in_out_len`

开放/新增更多vLLM推理参数，主要用于sampler测试、固定输入/输出长度下的测试。新增参数如下：

|              参数名称               |                                              说明                                              |
|------------------------------------|------------------------------------------------------------------------------------------------|
|             --top-p                | Float that controls the cumulative probability of the top tokens to consider. |
|             --top-k                | Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.  |
|           --ignore-eos             | Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.  |
|        --presence-penalty          | Float that penalizes new tokens based on whether they appear in the generated text so far.  |
|       --frequency-penalty          | Float that penalizes new tokens based on their "frequency in the generated text so far.  |
|       --repetition-penalty         | Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.  |
|    --include-stop-str-in-output    | Whether to include the stop strings in output text. Defaults to False. only support for vllm backend  |
|       --keep-special-tokens        | Whether to keep special tokens in the output. Defaults to False. only support for vllm backend  |
|       --strict-in-out-len          |  Whether to strictly enforce input output length. Defaults to False. only support for vllm backend |

### 大语言模型测试

server端启动
```shell
python3 -m vllm.entrypoints.openai.api_server \
        --model=[path of model] \
        --tensor-parallel-size=1 \
        --max-model-len=8192 \
        --dtype=float16 \
        --disable-log-requests
```
各参数含义如下：
- --model: 模型存储路径
- --tensor-parallel-size: 张量并行卡数，默认值为1，可按需调整
- --max-model-len: 模型支持的最大sequence长度
- --dtype: 推理精度，默认为`auto`，可按需调整
- --disable-log-requests: 不输出请求的日志信息

client端启动
```shell
python3 -m vllm_utils.benchmark_serving \
        --backend vllm \
        --dataset-name random \
        --model /pretrained_models/Yi-34B-Chat-w8a16 \
        --num-prompts 10 \
        --random-input-len 1024 \
        --random-output-len 512 \
        --temperature 0.5 \
        --top_p 0.5 \
        --top_k 50 \
        --presence_penalty 2.0 \
        --frequency_penalty 2.0 \
        --repetition_penalty 2.0 \
        --ignore-eos \
        --keep-special-tokens \
        --include-stop-str-in-output \
        --strict-in-out-len
```
各参数含义如下：
- --backend: 模型推理backend，默认为vllm
- --dataset-name: 测试用的数据集，可选项：`sharegpt`, `sonnet`, `random`
- --model: 模型存储路径
- --num-prompts: 总共发送的请求数，默认为1000，可按需调整
- --random-input-len: 使用random数据集时，模型的输入长度
- --random-output-len: 使用random数据集时，模型的输出长度
- --temperature: 控制采样随机性，平滑模型输出
- --top_p: 控制采样随机性，输出前top_p累计概率的token
- --top_k: 控制采样随机性，输出前top_k个token
- --presence_penalty: 控制采样随机性，输出token重复出现的惩罚（仅在输出token内检查）
- --frequency_penalty: 控制采样随机性，输出token重复频率的惩罚
- --repetition_penalty: 控制采样随机性，输出token重复出现的惩罚（在输出token和输入prompt内检查）
- --ignore-eos: 是否忽略终止符，继续生成token
- --keep-special-tokens: 是否将特殊字符返回给client端
- --include-stop-str-in-output: 是否将终止符返回给client端
- --strict-in-out-len: 是否严格控制input/output长度，最大程度保证实际input/output和测试参数一致
