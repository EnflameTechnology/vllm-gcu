## 大语言模型功能及性能测试
`benchmark_test`是Enflame新增的测试入口，使用方式为`python3 -m vllm_utils.benchmark_test arg1 arg2 ...`，具体可参考第5章各模型的测试命令。

功能上，支持：
- 使用内置prompt进行离线推理以展示LLM的推理能力；
- 使用指定长度的伪prompt进行LLM的推理性能测试；

可以通过`python3 -m vllm_utils.benchmark_test --help`查看各参数含义。

### 离线推理测试
离线推理测试，是使用`benchmark_test`内置prompt进行推理，用于演示在Enflame gcu上可以运行指定LLM的推理。

推理时，可通过`--demo`参数设置使用的prompt类型，`--demo=te/tc/ch`设置使用英文/中文/chat类型prompt，其余类型prompt类型及对应参数可通过`python3 -m vllm_utils.benchmark_test --help`查看。

### 性能测试
性能测试，是使用自动生成的、由多个`hi`组成的输入prompt进行推理，过程中忽略停止字符，生成指定长度的输出`token`。推理完成后，统计`TPS(Tokens per second)`、`TTFT(Time to first token)`等指标，以验证Enflame gcu的推理性能。

输入prompt长度由`--input-len`参数设定，输出token数由`--output-len`参数设定。其余相关参数按需设置。

测试完成输出的结果中，`latency_num_prompts`表示本轮推理的总耗时，`latency_per_token`表示每个输出token的latency，`request_per_second`表示以request为单位计算的吞吐，`token_per_second`表示以token为单位计算的吞吐，`prefill_latency_per_token`表示prefill阶段各token的latency，`decode_latency_per_token`表示decode阶段各token的latency，`decode_throughput`表示decode阶段的吞吐。

注意，获取`TTFT`指标时，需设置`--output-len=1`，推理完成后输出的`latency_per_token`即为`TTFT`。