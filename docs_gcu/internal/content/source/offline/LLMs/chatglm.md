## chatglm2/3


### chatglm2-6b
#### 模型下载
从huggingface上下载下列任意模型的预训练ckpt，路径记为[path of chatglmckpt]

- [chatglm2-6b](https://huggingface.co/THUDM/chatglm2-6b/commit/7fabe56db91e085c9c027f56f1c654d137bdba40)

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglmckpt] \
 --demo=te \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglmckpt] \
 --input-len=128 \
 --output-len=32640 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16 \
 --max-model-len=32768
```

注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

* chatglm2/3 32k模型运行时需要添加环境变量：
  * `export PYTORCH_GCU_ALLOC_CONF=backend:topsMallocAsync`

#### 基于OpenCompass进行mmlu数据集评测

1. 安装OpenCompass

执行 [OpenCompass的安装步骤](https://github.com/open-compass/opencompass/blob/main/README_zh-CN.md#%EF%B8%8F-%E5%AE%89%E8%A3%85)

注：建议使用OpenCompass0.3.1版本。如果安装依赖时安装了和torch_gcu不一致的版本，请重新手动安装。

注：需要安装以下依赖：

```shell
python3 -m pip install opencv-python==4.9.0.80
python3 -m pip install huggingface-hub==0.25.2
# for x86_64
python3 -m pip install torchvision==0.21.0+cpu -i https://download.pytorch.org/whl/cpu
# for aarch64
python3 -m pip install torchvision==0.21.0
# for x86_64 and python_version>=3.10
python3 -m pip install importlib-metadata==8.5.0
# for aarch64 and python_version>=3.10
python3 -m pip install importlib-metadata==4.6.4
```

2. 准备config文件

将下面的配置信息存为一个python文件，放入OpenCompass中如下路径`configs/models/chatglm/vllm_chatglm2_6b.py`

```python
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='chatglm2-6b-vllm',
        path='/path/to/chatglm2-6b',
        max_out_len=100,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', enforce_eager=True)
    )
]

```

执行以下命令

```
python3 run.py \
 --models=vllm_chatglm2_6b \
 --datasets=mmlu_gen
```

### chatglm3-6b-128k
#### 模型下载

*  url: [chatglm3-6b-128k](https://huggingface.co/THUDM/chatglm3-6b-128k)

*  branch: `main`

*  commit id: `6381a64`

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglm3-6b-128k] \
 --tensor-parallel-size 1 \
 --max-model-len=131072 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --gpu-memory-utilization 0.945
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglm3-6b-128k] \
 --tensor-parallel-size 1 \
 --max-model-len=131072 \
 --input-len=2048 \
 --output-len=2048 \
 --dtype=float16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64 \
 --gpu-memory-utilization 0.945
```

注：
*  本模型支持的`max-model-len`为131072；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;