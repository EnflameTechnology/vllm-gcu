## deepseek

### deepseek-moe-16b-base
本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [deepseek-moe-16b-base](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main)

*  branch: `main`

*  commit id: `521d2bc`

将上述url设定的路径下的内容全部下载到`deepseek-moe-16b-base`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of deepseek-moe-16b-base] \
 --demo=te \
 --output-len=256 \
 --dtype=float16
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of deepseek-moe-16b-base] \
 --tokenizer=[path of deepseek-moe-16b-base] \
 --max-model-len=1024 \
 --input-len=512 \
 --output-len=512 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;