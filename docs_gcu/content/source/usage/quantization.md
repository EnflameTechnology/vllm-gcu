## 量化
### 功能介绍

vLLM-gcu中，支持如下量化算法：
* GPTQ
* AWQ
* INT8 (W8A8)
* INT8 (W8A16)
* INT8 KVCache

注：
* 对于4bit per group量化方法(GPTQ和AWQ)，支持group size是64或64的整数倍。
* 对于GPTQ量化方法，当前暂不支持g_idx是乱序情况。
* INT8 (W8A8)量化方法当前支持权重为per-channel粒度，激活为per-tensor粒度。
* INT8 KVCache可以与其他量化方法正交使用。
* 对于原生vllm支持的squeezellm，gptq_marlin，fp8等方法gcu暂不支持。

### 量化过程

可以参考《TopsCompressor用户使用手册》进行模型量化及导出。

### 使用方法

以vllm_utils.benchmark_test为例：

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of quantized model] \
 --demo=te \
 --dtype=float16 \
 --quantization=[quantization method] \
 --output-len=256
```
注：
* `model`参数指定量化模型checkpoint本地地址；
* `quantization`参数为对应量化方法，可以如下字段：`gptq`，`awq`，`w8a16`，`w8a8`。

对于包含INT8 KVCache量化模型：
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of quantized model] \
    --tokenizer=[path of model tokenizer] \
    --output-len=128 \
    --device=gcu \
    --dtype=float16 \
    --quantization=[quantization method] \
    --quantization-param-path=[path of int8_kv_cache.json] \
    --kv-cache-dtype=int8
```
注：
* `model`参数指定量化模型checkpoint本地地址；
* `quantization`参数为对应量化方法，可以如下字段：`gptq`，`awq`，`w8a16`，`w8a8`；
* `quantization-param-path`参数指定量化模型int8 kvcache参数文件地址；
* `kv-cache-dtype`参数指定int8。