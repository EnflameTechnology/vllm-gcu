## SUS-Chat
### SUS-Chat-34B-w8a16_gptq

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`SUS-Chat-34B-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`SUS-Chat-34B-w8a16_gptq`文件夹中。
- `SUS-Chat-34B-w8a16_gptq`目录结构如下所示：

```shell
SUS-Chat-34B-w8a16_gptq/
    ├── config.json
    ├── config_tmp.json
    ├── model.safetensors
    ├── quantize_config.json
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── tokenizer.model
    └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo="te" \
    --model=[path of SUS-Chat-34B-w8a16_gptq] \
    --output-len=256 \
    --dtype=float16 \
    --tensor-parallel-size 2 \
    --device gcu\
    --quantization=gptq
```

#### serving模式
```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
    --model=[path of SUS-Chat-34B-w8a16_gptq] \
    --dtype float16 \
    --quantization gptq \
    --trust-remote-code \
    --max-model-len 4096 \
    --device gcu

# 启动客户端
python3 -m vllm_utils.benchmark_serving \
    --backend vllm \
    --dataset-name random \
    --model=[path of SUS-Chat-34B-w8a16_gptq] \
    --num-prompts 10 \
    --random-input-len 4 \
    --random-output-len 300 \
    --trust-remote-code
```
注：
*  为保证输入输出长度固定，数据集使用随机数测试；

*  num-prompts, random-input-len和random-output-len可按需调整；
