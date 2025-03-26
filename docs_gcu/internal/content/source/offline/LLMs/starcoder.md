## starcoder

### starcoderbase

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [starcoderbase](https://huggingface.co/bigcode/starcoderbase)
* branch: `main`
* commit id: `88ec5781ad071a9d9e925cd28f327dea22eb5188`


#### 批量离线推理
```shell
# completion
python3 -m vllm_utils.benchmark_test \
 --model=[path of starcoderbase] \
 --demo=ci \
 --template=templates/template_starcoder_completion.jinja \
 --dtype=float16 \
 --output-len=256
# infilling
python3 -m vllm_utils.benchmark_test \
 --model=[path of starcoderbase] \
 --demo=ci \
 --template=templates/template_starcoder_infilling.jinja \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of starcoderbase] \
 --max-model-len=8192 \
 --tokenizer=[path of starcoderbase] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为8192；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### starcoder2-7b

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [starcoderbase](https://huggingface.co/bigcode/starcoder2-7b/tree/main)
* branch: `main`
* commit id: `a3d3368`


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of starcoder2-7b] \
 --demo=cc \
 --template=templates/template_starcoder2_completion.jinja \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of starcoder2-7b] \
 --max-model-len=1024 \
 --tokenizer=[path of starcoder2-7b] \
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
