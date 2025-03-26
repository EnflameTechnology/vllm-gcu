## Orion

### Orion-14B-Base

#### 模型下载
*  url: [Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Base/tree/main)

*  branch: `main`

*  commit id: `b988a51`

将上述url设定的路径下的内容全部下载到`orion-14b-base`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of orion-14b-base] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of orion-14b-base] \
 --max-model-len=1024 \
 --tokenizer=[path of orion-14b-base] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096;
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Orion-14B-Chat

#### 模型下载
*  url: [Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Chat/tree/main)

*  branch: `main`

*  commit id: `7aa75f1`

将上述url设定的路径下的内容全部下载到`orion-14b-chat`文件夹中。

#### requirements

```shell
python3 -m pip install Jinja2==3.1.3 
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of orion-14b-chat] \
 --demo=ch \
 --template=default \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of orion-14b-chat] \
 --max-model-len=1024 \
 --tokenizer=[path of orion-14b-chat] \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=8 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096;
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
