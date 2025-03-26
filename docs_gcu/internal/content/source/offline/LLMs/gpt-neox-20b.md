## gpt-neox

### gpt-neox-20b
本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
*  url: [gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)

*  branch: `main`

*  commit id: `9369f14`

将上述url设定的路径下的内容全部下载到`gpt-neox-20b`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of hf_gpt-neox-20b_model] \
 --tensor-parallel-size=2 \
 --demo=te \
 --dtype=float16 \
 --output-len=20 \
 --gpu-memory-utilization=0.945
```

#### 性能测试
```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of hf_gpt-neox-20b_model] \
 --tensor-parallel-size=2 \
 --max-model-len=2048 \
 --tokenizer=[path of hf_gpt-neox-20b_model] \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=64 \
 --block-size=64 \
 --dtype=float16 \
 --gpu-memory-utilization=0.945
```
注：
*  本模型支持的`max-model-len`为2048；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;
