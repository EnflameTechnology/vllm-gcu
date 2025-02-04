## moose

### moose-34b_w8a16_gptq
本模型推理及性能测试至少需要2张enflame gcu。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --demo="te" \
 --model=[path of moose-34b_w8a16_gptq] \
 --output-len=256 \
 --dtype=float16 \
 --tensor-parallel-size 2 \
 --quantization=gptq \
 --device gcu
```

#### serving模式
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of moose-34b_w8a16_gptq] \
 --dtype float16 \
 --quantization gptq \
 --trust-remote-code \
 --max-model-len 4096 \
 --device gcu \
 --tensor-parallel-size 2 \
 --disable-custom-all-reduce \
 --disable-frontend-multiprocessing

# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model=[path of moose-34b_w8a16_gptq] \
 --num-prompts 10 \
 --random-input-len 4  \
 --random-output-len 300 \
 --trust-remote-code
```
注：
* 本模型支持的`max-model-len`为32k；
* 为保证输入输出长度固定，数据集使用随机数测试；
* `num-prompts`, `random-input-len`和`random-output-len`可按需调整；