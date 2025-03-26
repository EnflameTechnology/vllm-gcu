## dbrx

### dbrx-instruct

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [dbrx-instruct](https://huggingface.co/databricks/dbrx-instruct/tree/main)

*  branch: `main`

*  commit id: `c0a9245`

将上述url设定的路径下的内容全部下载到`dbrx-instruct`文件夹中。


#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of dbrx-instruct] \
 --output-len=256 \
 --demo=te \
 --dtype=float16 \
 --tensor-parallel-size=8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of dbrx-instruct] \
 --max-model-len=4096 \
 --tokenizer=[path of dbrx-instruct] \
 --input-len=1024 \
 --output-len=3972 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --tensor-parallel-size=8
```
注：
*  本模型支持的`max-model-len`为32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  `dtype`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;