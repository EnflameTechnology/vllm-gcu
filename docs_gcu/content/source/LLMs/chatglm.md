## chatglm2/3

chatglm系列模型，使用vllm 0.6.1.post2及以上版本时，需要手动降级transformers库版本

```
python3 -m pip install transformers==4.43.0
```

### chatglm2/3-6b
#### 模型下载
从huggingface上下载下列任意模型的预训练ckpt，路径记为[path of chatglmckpt]

- [chatglm2-6b-32k](https://huggingface.co/THUDM/chatglm2-6b-32k/commit/a2065f5dc8253f036a209e642d7220a942d92765)
- [chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b/commit/e46a14881eae613281abbd266ee918e93a56018f)
- [chatglm3-6b-32k](https://huggingface.co/THUDM/chatglm3-6b-32k/commit/e210410255278dd9d74463cf396ba559c0ef801c)

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
*  本模型支持的`max-model-len`为8192(chatglm3-6b ckpt)/32768(chatglm2-6b-32k和chatglm3-6b-32k ckpt)；

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

将下面的配置信息存为一个python文件，放入OpenCompass中如下路径`configs/models/chatglm/vllm_chatglm3_6b.py`

```python
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='chatglm3-6b-vllm',
        path='/path/to/chatglm3-6b',
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
 --models=vllm_chatglm3_6b \
 --datasets=mmlu_gen
```

### chatglm2/3-6b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载 `ChatGLM2-6b-8k-w8a16_gptq.tar` 或  `chatglm2-6b-32k-w8a16_gptq.tar` 或  `ChatGLM3-6b-8k-w8a16_gptq.tar` 或  `ChatGLM3-6b-32k-w8a16_gptq.tar` 文件并解压，将压缩包内的内容全部拷贝到`chatglm_w8a16_gptq`文件夹中。
- `chatglm_w8a16_gptq`目录结构如下所示：

```shell
chatglm_w8a16_gptq/
├── config.json
├── configuration_chatglm.py
├── model.safetensors
├── quantize_config.json
├── tokenization_chatglm.py
├── tokenizer_config.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chatglm_w8a16_gptq] \
 --demo=tc \
 --dtype=float16 \
 --quantization=gptq \
 --output-len=256 \
 --trust-remote-code
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chatglm_w8a16_gptq] \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=16 \
 --block-size=64 \
 --dtype=float16 \
 --quantization=gptq \
 --max-model-len 8192 \
 --trust-remote-code
```
