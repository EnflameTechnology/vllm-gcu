## chatglm3

chatglm系列模型，使用vllm 0.6.1.post2及以上版本时，需要手动降级transformers库版本

```
python3 -m pip install transformers==4.43.0
```

### chatglm3-6b
#### 模型下载
从huggingface上下载下列任意模型的预训练ckpt，路径记为[path of chatglmckpt]

- [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b/commit/e46a14881eae613281abbd266ee918e93a56018f)
- [chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k/commit/e210410255278dd9d74463cf396ba559c0ef801c)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=28672 \
 --output-len=4096 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16 \
 --max-model-len=32768 \
 --device "gcu" \
 --tokenizer "chatglm3-6b-32k"
```

注：
*  本模型支持的`max-model-len`为8192(chatglm3-6b ckpt)/32768(chatglm3-6b-32k ckpt)；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

* chatglm3 32k模型运行时需要添加环境变量：
  * `export PYTORCH_GCU_ALLOC_CONF=backend:topsMallocAsync`

### chatglm3-6b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载 `ChatGLM3-6b-8k-w8a16_gptq.tar` 或  `ChatGLM3-6b-32k-w8a16_gptq.tar` 文件并解压，将压缩包内的内容全部拷贝到`chatglm_w8a16_gptq`文件夹中。
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
 --demo=te \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=256 \
 --disable-log-stats
```

