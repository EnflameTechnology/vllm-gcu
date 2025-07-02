## glm4

### glm-4-9b-chat

#### 模型下载
从huggingface上下载下列任意模型的预训练ckpt，路径记为[path of chatglmckpt]

- [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat/tree/b84dc74294ccd507a3d78bde8aebf628221af9bd)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=tc \
 --output-len=256 \
 --max-model-len=33792
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=32768 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --max-model-len=33792 \
 --trust-remote-code
```

注：
*  glm-4-9b-chat模型支持的`max-model-len`为128k，gcu单卡当前支持到33792；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### GLM-Z1-32B-0414-GPTQ-Int4

#### 模型下载
*  url: [GLM-Z1-32B-0414-GPTQ-Int4](https://www.modelscope.cn/models/tclf90/glm-z1-32b-0414-gptq-int4/files)

*  branch: `master`

*  commit id: `ec6ecf2793f061005011f078e8ae0975bcb5ace8`

将上述url设定的路径下的内容全部下载到`GLM-Z1-32B-0414-GPTQ-Int4`文件夹中。

注：需要安装以下依赖：

```shell
python3 -m pip install transformers==4.51.3
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of GLM-Z1-32B-0414-GPTQ-Int4] \
 --demo=te \
 --dtype=float16 \
 --quantization=gptq_gcu \
 --tensor-parallel-size=1 \
 --output-len=128 \
 --block-size=64 \
 --max-model-len=32768 \
 --gpu-memory-utilization=0.9 \
 --trust-remote-code \
 --device gcu
```

#### serving模式

```shell
# 启动服务端
  python3 -m vllm.entrypoints.openai.api_server \
  --model [path of GLM-Z1-32B-0414-GPTQ-Int4] \
  --num-scheduler-steps=16 \
  --tensor-parallel-size 1 \
  --max-seq-len-to-capture=32768 \
  --max-model-len 32768 \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --block-size=64 \
  --dtype=float16 \
  --quantization=gptq_gcu


# 启动客户端
  python3 -m vllm_utils.benchmark_serving \
  --backend vllm \
  --dataset-name random \
  --model [path of GLM-Z1-32B-0414-GPTQ-Int4] \
  --num-prompts 1 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --trust-remote-code \
  --ignore_eos \
  --strict-in-out-len \
  --keep-special-tokens
```

注：
*  GLM-Z1-32B-0414-GPTQ-Int4模型支持的`max-model-len`为32k；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;