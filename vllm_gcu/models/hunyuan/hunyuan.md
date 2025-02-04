## hunyuan

### hunyuan-moe-3b
本模型推理及性能测试至少需要2张enflame gcu。

#### 模型结构文件下载
* 联系Enflame通过EGC下载所需的OOT_models文件夹，保存到测试机上；

#### 性能测试
* 添加环境变量 export VLLM_OOT_MODEL_PATH=[path of OOT_models]
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of hunyuan-moe-3b]     \
 --tokenizer=[path of hunyuan-moe-3b] \
 --tensor-parallel-size=4 \
 --max-model-len=4096     \
 --input-len=1500  \
 --output-len=256  \
 --num-prompts=1   \
 --block-size=64   \
 --dtype=bfloat16
```
注：
*  本模型支持的`max-model-len`为34k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### hunyuan-moe-3b-w8a8c8
本模型推理及性能测试至少需要1张enflame gcu。

#### 模型结构文件下载
* 联系Enflame通过EGC下载所需的OOT_models文件夹，保存到测试机上；

#### serving模式
* 添加环境变量 export VLLM_OOT_MODEL_PATH=[path of OOT_models]
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server --model=[path of hunyuan-moe-3b-w8a8c8]  \
 --quantization-param-path=[path of hunyuan-moe-3b-w8a8c8/int8_kv_cache.json]          \
 --kv-cache-dtype=int8  \
 --max-model-len=34816  \
 --disable-log-requests \
 --tensor-parallel-size=4      \
 --gpu-memory-utilization=0.945  \
 --block-size=64 \
 --dtype=float16 \
 --trust-remote-code

# 启动客户端
python3 -m vllm_utils.benchmark_serving --backend=vllm  \
 --dataset-name=random  \
 --model=[path of hunyuan-moe-3b-w8a8c8]  \
 --num-prompts=1  \
 --random-input-len=4000   \
 --random-output-len=256   \
 --trust-remote-code
```
注：
* 为保证输入输出长度固定，数据集使用随机数测试；
* num-prompts, random-input-len和random-output-len可按需调整；

### huanyuan-moe-3b-w8a16_gptq

#### 模型结构文件下载
* 联系Enflame通过EGC下载所需的OOT_models文件夹，保存到测试机上；

#### 性能测试
* 添加环境变量 export VLLM_OOT_MODEL_PATH=[path of OOT_models]
```shell
python3 -m vllm_utils.benchmark_test \
 --perf \
 --model=[path of huanyuan-moe-3b-w8a16_gptq] \
 --tokenizer=[path of huanyuan-moe-3b-w8a16_gptq] \
 --input-len 3500 \
 --output-len 256 \
 --dtype=float16 \
 --tensor-parallel-size=4 \
 --max-model-len=4096 \
 --num-prompts=1 \
 --block-size=64 \
 --quantization gptq \
 --gpu-memory-utilization=0.945
```
注：
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;