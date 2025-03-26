## WizardCoder

### WizardCoder-15B-V1.0

#### 模型下载
*  url: [WizardCoder-15B-V1.0](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0/tree/main)

*  branch: `main`

*  commit id: `9c17758`

将上述url设定的路径下的内容全部下载到`wizardcoder-15b-v1.0`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of wizardcoder-15b-v1.0] \
 --output-len=20 \
 --demo=cin \
 --dtype=float16 \
 --template=templates/template_wizardcoder_instruct.jinja
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of wizardcoder-15b-v1.0] \
 --max-model-len=1024 \
 --tokenizer=[path of wizardcoder-15b-v1.0] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为8k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;