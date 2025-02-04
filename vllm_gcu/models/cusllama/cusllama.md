## cusllama

### llama2-130b-w8a16_gptq
本模型推理及性能测试需要8张enflame gcu。

#### 模型结构文件下载
* 联系Enflame通过EGC下载所需的OOT_models文件夹，保存到测试机上；

#### 性能测试
* 添加环境变量 export VLLM_OOT_MODEL_PATH=[path of OOT_models]
```shell
python3 -m vllm_utils.benchmark_test  --perf \
 --model=[path of llama2-130b-w8a16_gptq] \
 --device=gcu \
 --max-model-len=8192 \
 --tokenizer=[path of llama2-130b-w8a16_gptq] \
 --input-len=2048 \
 --output-len=4096 \
 --num-prompts=1 \
 --tensor-parallel-size=8 \
 --quantization=gptq \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama2-130b-w8a16c8
本模型推理及性能测试需要8张enflame gcu。

#### 模型结构文件下载
* 联系Enflame通过EGC下载所需的OOT_models文件夹，保存到测试机上；

#### 性能测试
* 添加环境变量 export VLLM_OOT_MODEL_PATH=[path of OOT_models]
```shell
python3 -m vllm_utils.benchmark_test  --perf \
 --model=[path of llama2-130b-w8a16_gptq] \
 --device=gcu \
 --max-model-len=128128 \
 --input-len=128000 \
 --output-len=128 \
 --num-prompts=1 \
 --tensor-parallel-size=8 \
 --quantization=gptq \
 --block-size=64 \
 --dtype=float16 \
 --quantization-param-path [path of int8_kv_cache.json] \
 --kv-cache-dtype int8
```
注：
*  本模型支持的`max-model-len`为256130，需要8张卡跑128128；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;