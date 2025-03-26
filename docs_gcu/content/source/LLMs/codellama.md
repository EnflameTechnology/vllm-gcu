## codellama

### codellama-70b-instruct-w8a16_gptq

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`codellama-70b-instruct-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`codellama_70b_instruct_w8a16_gptq`文件夹中。
- `codellama_70b_instruct_w8a16_gptq`目录结构如下所示：

```shell
codellama_70b_instruct_w8a16_gptq/
            ├── added_tokens.json
            ├── config.json
            ├── generation_config.json
            ├── model.safetensors
            ├── quantize_config.json
            ├── special_tokens_map.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── tokenizer.model
            └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of codellama_70b_instruct_w8a16_gptq] \
 --tensor-parallel-size=2 \
 --demo=cc \
 --output-len=256 \
 --dtype=float16  \
 --quantization gptq
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of codellama_70b_instruct_w8a16_gptq] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of codellama_70b_instruct_w8a16_gptq] \
 --input-len=1024 \
 --output-len=3072 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
