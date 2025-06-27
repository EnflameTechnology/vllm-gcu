## glm

### glm-4-9b

#### 模型下载
从huggingface上下载模型的预训练ckpt，路径记为[path of chatglmckpt]

- [glm-4-9b](https://huggingface.co/THUDM/glm-4-9b/tree/e6efe95a013d4d3d0a9c7d71e89d117e860dc2f3)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=tc \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=1024 \
 --output-len=2048 \
 --num-prompts=16 \
 --block-size=64 \
 --enforce-eager \
 --trust-remote-code
```

注：
*  glm-4-9b模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;