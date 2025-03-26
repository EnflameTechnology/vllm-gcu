## opt

### opt-13b
#### 模型下载
*  url:[opt-13b](https://huggingface.co/facebook/opt-13b/tree/main)

*  branch:`main`

*  commit id:`e515202`

将上述url设定的路径下的内容全部下载到`opt-13b`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test --disable-log-stats  \
 --max-model-len=2048 \
 --model=[path of opt-13b] \
 --tokenizer=[path of opt-13b] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of opt-13b] \
 --max-model-len=2048 \
 --tokenizer=[path of opt-13b] \
 --input-len=128 \
 --output-len=128 \
 --dtype=float16 \
 --num-prompts=1 \
 --block-size=64
```
注：
*  本模型支持的`max-model-len`2048

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
