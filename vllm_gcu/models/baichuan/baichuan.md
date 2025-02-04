## baichuan2

### baichuan2-33b-w4a16c8
本模型推理及性能测试至少需要4张enflame gcu。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
--model [path of baichuan2_33b_w4a16c8] \
--tensor-parallel-size 4 \
--output-len=512 --demo=te --dtype=float16 \
--device gcu \
--quantization=gptq  \
--kv-cache-dtype="int8" \
--quantization-param-path=[path of int8_kv_cache.json]
```

#### 性能测试
```shell
 python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of baichuan2_33b_w4a16c8] \
 --device=gcu \
 --tensor-parallel-size=4 \
 --tokenizer=[path of baichuan2_33b_w4a16c8] \
 --input-len=1024 --output-len=512 \
 --num-prompts=1 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq  \
 --kv-cache-dtype="int8" \
 --quantization-param-path=[path of int8_kv_cache.json] \
 --gpu-memory-utilization=0.945
```
注：
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-33b-w8a8c8
本模型推理及性能测试至少需要4张enflame gcu。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
--model [path of baichuan2_33b_w8a8c8] \
--tensor-parallel-size 4 \
--output-len=128 --demo=tc --dtype=float16 \
--device gcu \
--quantization=w8a8  \
--kv-cache-dtype="int8" \
--quantization-param-path=[path of int8_kv_cache.json]
```

#### 性能测试
```shell
 python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of baichuan2_33b_w8a8c8] \
 --device=gcu \
 --tensor-parallel-size=4 \
 --tokenizer=[path of baichuan2_33b_w8a8c8] \
 --input-len=1024 --output-len=512 \
 --num-prompts=1 --block-size=64 \
 --dtype=float16 \
 --quantization=w8a8  \
 --kv-cache-dtype="int8" \
 --quantization-param-path=[path of int8_kv_cache.json]
```
注：
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-33b
本模型推理及性能测试需要4张enflame gcu。

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of baichuan2-33b] \
 --device=gcu \
 --max-model-len=2048 \
 --tokenizer=[path of baichuan2-33b] \
 --input-len=1024 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --tensor-parallel-size 4
```
注：
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### baichuan2-33b-w8a16_gptq
本模型推理及性能测试需要2张enflame gcu。

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test \
       --perf \
       --model=[path of baichuan2-33b-w8a16_gptq] \
       --device=gcu \
       --max-model-len=2048 \
       --tokenizer=[path of baichuan2-33b-w8a16_gptq] \
       --input-len=1024 \
       --output-len=512 \
       --num-prompts=1 \
       --block-size=64 \
       --tensor-parallel-size 2 \
       --quantization gptq
```
注：
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
