## gpt-j

### gpt-j-6b
#### 模型下载
*  url: [gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b)

*  branch:`main`

*  commit id:`47e1693`

将上述url设定的路径下的内容全部下载到`gpt-j-6b`文件夹中。
#### merges.txt 文件下载
*  url:[merges.txt](https://huggingface.co/openai-community/gpt2/blob/main/merges.txt)

*  branch:`main`

*  commit id:`a9eff68`

将merges.txt 文件放到`gpt-j-6b`文件夹中。
#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of gpt-j-6b] \
 --max-model-len=256 \
 --output-len=256 \
 --demo='te' \
 --dtype=float16
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of gpt-j-6b] \
 --max-model-len=2048 \
 --tokenizer=[path of gpt-j-6b] \
 --input-len=128 \
 --output-len=1920 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为2048;

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

