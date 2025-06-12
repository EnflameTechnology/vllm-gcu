## Ziya-Coding

### Ziya-Coding-34B-v1.0

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0)

*  branch: `main`

*  commit id: `deda16c`

将上述url设定的路径下的内容全部下载到`Ziya-Coding-34B-v1.0`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Ziya-Coding-34B-v1.0] \
 --tensor-parallel-size=2 \
 --max-model-len=16384 \
 --output-len=256 \
 --demo=cin \
 --dtype=float16 \
 --template=templates/template_ziya_instruct.jinja
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Ziya-Coding-34B-v1.0] \
 --tensor-parallel-size=2 \
 --max-model-len=12288 \
 --tokenizer=[path of Ziya-Coding-34B-v1.0] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=float16
```
注：
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Ziya-Coding-34B-v1.0-w8a16_gptq

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Ziya-Coding-34B-v1.0-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`Ziya_Coding_34B_v1.0_w8a16_gptq`文件夹中。
- `Ziya_Coding_34B_v1.0_w8a16_gptq`目录结构如下所示：

```shell
Ziya_Coding_34B_v1.0_w8a16_gptq/
      ├── config.json
      ├── model.safetensors
      ├── quantize_config.json
      ├── special_tokens_map.json
      ├── tokenizer_config.json
      ├── tokenizer.model
      └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Ziya_Coding_34B_v1.0_w8a16_gptq] \
 --max-model-len=8192 \
 --output-len=256 \
 --demo=cin \
 --dtype=float16 \
 --quantization gptq \
 --template=templates/template_ziya_instruct.jinja
```
注：
* 单张gcu上可以支持的`max-model-len`为8192，若需使用到模型自身支持的16384的`max-model-len`，则需设置`--tensor-parallel-size=2`；

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Ziya_Coding_34B_v1.0_w8a16_gptq] \
 --max-model-len=8192 \
 --tokenizer=[path of Ziya_Coding_34B_v1.0_w8a16_gptq] \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --quantization=gptq \
 --dtype=float16
```

注:
* 单张gcu上可以支持的`max-model-len`为8192，若需使用到模型自身支持的16384的`max-model-len`，则需设置`--tensor-parallel-size=2`；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
