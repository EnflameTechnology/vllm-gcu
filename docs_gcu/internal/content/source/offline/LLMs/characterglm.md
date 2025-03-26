## CharacterGLM

### CharacterGLM-6B

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [CharacterGLM-6B](https://www.modelscope.cn/THUCoAI/CharacterGLM-6B.git)
* branch: `master`
* commit id: `e05e39c1bd66de62e2d501cd6f75a5fd1d9e0905`

- 将上述url设定的路径下的内容全部下载到`characterglm-6b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of characterglm-6b] \
 --demo=chc \
 --template=templates/template_characterglm_chat.jinja \
 --output-len=20 \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of characterglm-6b] \
 --max-model-len=32768 \
 --tokenizer=[path of characterglm-6b] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```

注:
* 本模型支持的`max-model-len`为32768
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
