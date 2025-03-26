## starcoder

### starcoder2-15b

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [starcoderbase](https://huggingface.co/bigcode/starcoder2-15b/tree/main)
* branch: `main`
* commit id: `995200d`


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of starcoder2-15b] \
 --demo=cc \
 --template=templates/template_starcoder2_completion.jinja \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of starcoder2-15b] \
 --max-model-len=1024 \
 --tokenizer=[path of starcoder2-15b] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为16384；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### starcoder2-15b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`starcoder2-15b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`starcoder2_15b_w8a16_gptq`文件夹中。
- `starcoder2_15b_w8a16_gptq`目录结构如下所示：

```shell
starcoder2_15b_w8a16_gptq/
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── special_tokens_map.json
            ├── merges.txt
            ├── vocab.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of starcoder2_15b_w8a16_gptq] \
 --demo=cc \
 --output-len=256 \
 --dtype=float16  \
 --template=templates/template_starcoder2_completion.jinja \
 --quantization gptq
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of starcoder2_15b_w8a16_gptq] \
 --max-model-len=16384 \
 --tokenizer=[path of starcoder2_15b_w8a16_gptq] \
 --input-len=8192 \
 --output-len=8192 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为16384；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;