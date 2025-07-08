## internlm

### internlm2_5-7b-chat-w4a16-gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`internlm2_5-7b-chat-w4a16-gptq-groupsize64.tar`文件并解压，将压缩包内的内容全部拷贝到`internlm2_5-7b-chat-w4a16-gptq`文件夹中。
- `internlm2_5-7b-chat-w4a16-gptq`目录结构如下所示：

```shell
internlm2_5-7b-chat-w4a16-gptq/
    ├── config.json
    ├── configuration_internlm2.py
    ├── generation_config.json
    ├── modeling_internlm2.py
    ├── model.safetensors
    ├── quantize_config.json
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenization_internlm2_fast.py
    ├── tokenization_internlm2.py
    ├── tokenizer_config.json
    ├── tokenizer.model
    └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=128 \
 --device gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --trust-remote-code \
 --tensor-parallel-size=1 \
 --tokenizer=[path of internlm2_5-7b-chat-w4a16-gptq] \
 --device gcu
```
注：
*  本模型支持的`max-model-len`为32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
