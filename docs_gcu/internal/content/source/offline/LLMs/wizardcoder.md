## WizardCoder

### WizardCoder-Python-34B-V1.0

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [WizardCoder-Python-34B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0/tree/main)

*  branch: `main`

*  commit id: `897fc6d`

将上述url设定的路径下的内容全部下载到`wizardcoder-python-34b-v1.0`文件夹中。

#### requirements

```shell
python3 -m pip install Jinja2==3.1.3 
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of wizardcoder-python-34b-v1.0] \
 --tensor-parallel-size=2 \
 --output-len=256 \
 --max-model-len=2048 \
 --demo=cin \
 --dtype=float16 \
 --template=templates/template_wizardcoder_instruct.jinja
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of wizardcoder-python-34b-v1.0] \
 --tensor-parallel-size=2 \
 --max-model-len=2048 \
 --tokenizer=[path of wizardcoder-python-34b-v1.0] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### WizardCoder-33B-V1.1

本模型推理及性能测试需要两张enflame gcu。

#### 模型下载
*  url: [WizardCoder-33B-V1.1](https://huggingface.co/WizardLM/WizardCoder-33B-V1.1/tree/main)

*  branch: `main`

*  commit id: `22d03e1`

将上述url设定的路径下的内容全部下载到`wizardcoder-33b-v1.1`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of wizardcoder-33b-v1.1] \
 --tensor-parallel-size=2 \
 --output-len=20 \
 --demo=cin \
 --dtype=float16 \
 --max-model-len=2048 \
 --template=templates/template_wizardcoder_instruct.jinja
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of wizardcoder-33b-v1.1] \
 --tensor-parallel-size=2 \
 --max-model-len=1024 \
 --tokenizer=[path of wizardcoder-33b-v1.1] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16k；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
