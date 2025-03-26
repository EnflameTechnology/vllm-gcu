## glm4

glm系列模型，使用vllm 0.6.1.post2及以上版本时，需要手动降级transformers库版本

```
python3 -m pip install transformers==4.43.0
```

### glm-4-9b

#### 模型下载
从huggingface上下载模型的预训练ckpt，路径记为[path of chatglmckpt]

- [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b/tree/e6efe95a013d4d3d0a9c7d71e89d117e860dc2f3)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=tc \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=1024 \
 --output-len=2048 \
 --num-prompts=16 \
 --block-size=64 \
 --enforce-eager \
 --trust-remote-code
```

注：
*  glm-4-9b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### glm-4-9b-chat

#### 模型下载
从huggingface上下载下列任意模型的预训练ckpt，路径记为[path of chatglmckpt]

- [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat/tree/b84dc74294ccd507a3d78bde8aebf628221af9bd)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=tc \
 --output-len=256 \
 --max-model-len=33792
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=32768 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=33792 \
 --trust-remote-code
```

注：
*  glm-4-9b-chat模型支持的`max-model-len`为128k，gcu单卡当前支持到33792；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### glm-4-9b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载 `glm-4-9b_W8A16.tar` 文件并解压，将压缩包内的内容全部拷贝到`chatglm_w8a16_gptq`文件夹中。
- `chatglm_w8a16_gptq`目录结构如下所示：

```shell
chatglm_w8a16_gptq/
├── config.json
├── configuration_chatglm.py
├── model.safetensors
├── quantize_config.json
├── tokenization_chatglm.py
├── tokenizer_config.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglm_w8a16_gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=256 \
 --max-model-len=8192
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglm_w8a16_gptq] \
 --input-len=512 \
 --output-len=2048 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --trust-remote-code \
 --enforce-eager
```

注：
*  glm-4-9b-w8a16_gptq模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
