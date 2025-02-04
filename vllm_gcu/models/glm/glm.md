## glm-4

glm系列模型，使用vllm 0.6.1.post2及以上版本时，需要手动降级transformers库版本

```
pip3 install transformers==4.43.0
```

### glm-4-32b w4a16c8

本模型推理及性能测试需要1张或更多enflame gcu，以4卡为例。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test --demo tc \
--model [path of glm-4-32b_w4a16] \
--device=gcu \
--tensor-parallel-size=4  \
--block-size=64 \
--trust-remote-code \
--kv-cache-dtype int8 \
--quantization gptq \
--quantization-param-path [path of int8_kv_cache.json] \
--dtype float16 \
--output-len 256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
--input-len 1024 \
--output-len 1024 \
--num-prompts 1 \
--model [path of glm-4-32b_w4a16] \
--quantization gptq \
--dtype float16 \
--device gcu \
--block-size 64 \
--trust-remote-code \
--gpu-memory-utilization 0.945 \
-tp 4 \
--kv-cache-dtype int8 \
--quantization-param [path of int8_kv_cache.json]
```

注：
*  glm-4-32b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### glm-4-130b w4a16c8

本模型推理及性能测试需要8张enflame gcu。

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
--input-len 1024 \
--output-len 1024 \
--num-prompts 1 \
--model [path of glm-4-130b-chat_w4a16] \
--quantization gptq \
--dtype float16 \
--device gcu \
--block-size 64 \
--trust-remote-code \
--gpu-memory-utilization 0.945 \
-tp 8 \
--kv-cache-dtype int8 \
--quantization-param [path of int8_kv_cache.json]
```

注：
*  glm-4-130b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### glm-4-32b w8a8c8

本模型推理及性能测试需要1张或更多enflame gcu，以4卡为例。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test --demo tc \
--model [path of glm-4-32b_w8a8] \
--device=gcu \
--tensor-parallel-size=4  \
--block-size=64 \
--trust-remote-code \
--kv-cache-dtype int8 \
--quantization w8a8 \
--quantization-param-path [path of int8_kv_cache.json] \
--dtype float16 \
--output-len 256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
--input-len 1024 \
--output-len 1024 \
--num-prompts 1 \
--model [path of glm-4-32b_w8a8] \
--quantization w8a8 \
--dtype float16 \
--device gcu \
--block-size 64 \
--trust-remote-code \
--gpu-memory-utilization 0.945 \
-tp 4 \
--kv-cache-dtype int8 \
--quantization-param [path of int8_kv_cache.json]
```

注：
*  glm-4-32b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### glm-4-130b w8a8c8

本模型推理及性能测试需要8张enflame gcu。

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
--input-len 1024 \
--output-len 1024 \
--num-prompts 1 \
--model [path of glm-4-130b-chat_w8a8] \
--quantization w8a8 \
--dtype float16 \
--device gcu \
--block-size 64 \
--trust-remote-code \
--gpu-memory-utilization 0.945 \
-tp 8 \
--kv-cache-dtype int8 \
--quantization-param [path of int8_kv_cache.json]
```

注：
*  glm-4-130b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

## glm-2

### chatglm2-32b-32k

本模型推理及性能测试需要2张enflame gcu。

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of chatglm2-32b-32k] \
 --max-model-len=2048  \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --tensor-parallel-size 2
```

注：
*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### chatglm2-32b-32k-w8a16_gptq

本模型推理及性能测试需要2张enflame gcu。

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of chatglm2-32b-32k-w8a16_gptq]  \
 --max-model-len=2048  \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --tensor-parallel-size 2 \
 --quantization gptq
```

注：
*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
