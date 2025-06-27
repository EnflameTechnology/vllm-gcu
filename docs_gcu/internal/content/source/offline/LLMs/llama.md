## llama

### llama-65b

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
* url: [llama-65b](https://github.com/facebookresearch/llama)
* branch: `llama_v1`
* commit id: `57b0eb62de0636e75af471e49e2f1862d908d9d8`

- 参考[download](https://github.com/facebookresearch/llama/tree/llama_v1?tab=readme-ov-file#download)下载llama-65b模型，将全部内容下载到`llama-65b`文件夹内。
- 参考[convert_llama_weights_to_hf.py](https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/convert_llama_weights_to_hf.py)，将下载的模型文件转为huggingface transformers格式，将转换的全部内容存放在`llama-65b-hf`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-65b-hf] \
 --demo=te \
 --tensor-parallel-size=4 \
 --dtype=float16 \
 --output-len=256
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-65b-hf] \
 --tensor-parallel-size=4 \
 --max-model-len=2048 \
 --tokenizer=[path of llama-65b-hf] \
 --input-len=128 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```

注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama2-13b

#### 模型下载
*  url:[llama2-13b](https://huggingface.co/meta-llama/Llama-2-13b-hf/tree/main)

*  branch:`main`

*  commit id:`638c8be`

将上述url设定的路径下的内容全部下载到`llama-2-13b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-2-13b-hf] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --gpu-memory-utilization=0.945
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-2-13b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of llama-2-13b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.945 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### chinese-llama-2-7b
#### 模型下载
*  url:[chinese-llama-2-7b](https://huggingface.co/hfl/chinese-llama-2-7b)

*  branch:`main`

*  commit id:`c40cf9a`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-7b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-7b-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-7b-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-7b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-7b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### chinese-llama-2-7b-16k
#### 模型下载
*  url:[chinese-llama-2-7b-16k](https://huggingface.co/hfl/chinese-llama-2-7b-16k)

*  branch:`main`

*  commit id:`c934a79`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-7b-16k-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-7b-16k-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-7b-16k-hf] \
 --dtype=float16 \
 --demo=tc \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-7b-16k-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-7b-16k-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为16384;

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### chinese-llama-2-13b

#### 模型下载
*  url:[chinese-llama-2-13b](https://huggingface.co/hfl/chinese-llama-2-13b)

*  branch:`main`

*  commit id:`043f8d2`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-13b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-13b-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-13b-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-13b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-13b-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### chinese-llama-2-13b-16k

#### 模型下载
*  url:[chinese-llama-2-13b-16k](https://huggingface.co/hfl/chinese-llama-2-13b-16k)

*  branch:`main`

*  commit id:`1c90d65`

将上述url设定的路径下的内容全部下载到`chinese-llama-2-13b-16k-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of chinese-llama-2-13b-16k-hf] \
 --max-model-len=256 \
 --tokenizer=[path of chinese-llama-2-13b-16k-hf] \
 --dtype=float16 \
 --demo=tc  \
 --output-len=20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of chinese-llama-2-13b-16k-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of chinese-llama-2-13b-16k-hf] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：

*  本模型支持的`max-model-len`为16384；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Meta-Llama-3.1-405B-Instruct-AWQ-INT4

#### 模型下载
*  url:[Meta-Llama-3.1-405B-Instruct-AWQ-INT4](https://www.modelscope.cn/models/llm-research/meta-llama-3.1-405b-instruct-awq-int4/)

*  branch:`master`

*  commit id:`02d8aeb3`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-405B-Instruct-AWQ-INT4`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3.1-405B-Instruct-AWQ-INT4] \
  --tensor-parallel-size 8 \
  --max-model-len=16384 \
  --output-len=512 \
  --demo=te \
  --dtype=float16 \
  --device gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-405B-Instruct-AWQ-INT4] \
 --tensor-parallel-size 8 \
 --max-model-len=16384 \
 --input-len=14336 \
 --output-len=1024 \
 --dtype=float16 \
 --device gcu \
 --num-prompts 1 \
 --block-size=64
```
注：

*  本模型支持的`max-model-len`为131072；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama2-7b
#### 模型下载
*  url:[llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main)

*  branch:`main`

*  commit id:`3f025b`

将上述url设定的路径下的内容全部下载到`llama-2-7b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-2-7b-hf] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-2-7b-hf] \
 --max-model-len=4096 \
 --tokenizer=[path of llama-2-7b-hf] \
 --input-len=3968 \
 --output-len=128 \
 --num-prompts=128 \
 --block-size=64 \
 --dtype=float16 \
 --gpu-memory-utilization=0.9
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

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

将下面的配置信息存为一个python文件，放入OpenCompass中如下路径`configs/models/llama/vllm_llama2_7b.py`

```python
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='llama2-7b-vllm',
        path='/path/to/Llama-2-7b-hf/',
        max_out_len=1,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu',
                          gpu_memory_utilization=0.7,
                          enforce_eager=True)
    )
]

```

3. 修改opencompass中的`opencompass/models/vllm.py`，增加get_ppl方法
```python
    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        assert mask_length is None, 'mask_length is not supported'
        bsz = len(inputs)

        # tokenize
        prompt_tokens = [self.tokenizer(x, truncation=True,
                                        add_special_tokens=False,
                                        max_length=self.max_seq_len - 1
                                        )['input_ids'] for x in inputs]
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_prompt_size)
        tokens = torch.zeros((bsz, total_len)).long()
        for k, t in enumerate(prompt_tokens):
            num_token = min(total_len, len(t))
            tokens[k, :num_token] = torch.tensor(t[-num_token:]).long()
        # forward
        generation_kwargs = {}
        generation_kwargs.update(self.generation_kwargs)
        global ce_loss, bz_idx
        ce_loss = []
        bz_idx = 0

        def logits_hook(logits):
            global ce_loss, bz_idx
            # compute ppl
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = tokens[bz_idx:bz_idx + logits.shape[0],
                                  1:logits.shape[1]
                                  ].contiguous().to(logits.device)
            bz_idx += logits.shape[0]
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss_fct = torch.nn.CrossEntropyLoss(
                reduction='none', ignore_index=0)
            loss = loss_fct(shift_logits, shift_labels).view(
                logits.shape[0], -1)
            lens = (shift_labels != 0).sum(-1).cpu().numpy()
            ce_loss.append(loss.sum(-1).cpu().detach().numpy() / lens)
            return logits

        generation_kwargs['full_logits_processors'] = [logits_hook]
        generation_kwargs['max_tokens'] = 1
        sampling_kwargs = SamplingParams(**generation_kwargs)
        outputs = self.model.generate(
            None, sampling_kwargs, prompt_tokens, use_tqdm=False)
        ce_loss = np.concatenate(ce_loss)
        return ce_loss
```

执行以下命令

```
python3 run.py \
 --models=vllm_llama2_7b \
 --datasets=mmlu_ppl
```


### llama2-7b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2-7b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`llama2-7b-w8a16_gptq`文件夹中。
- `llama2-7b-w8a16_gptq`目录结构如下所示：

```shell
llama2-7b-w8a16_gptq/
├── config.json
├── generation_config.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama2-7b-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --quantization gptq \
 --max-model-len=64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama2-7b-w8a16_gptq] \
 --max-model-len=4096 \
 --tokenizer=[path of llama2-7b-w8a16_gptq] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### llama2-7b-w4a16

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)
* branch: `main`
* commit id: `d5ad9310836dd91b6ac6133e2e47f47394386cea`

- 将上述url设定的路径下的内容全部下载到`Llama-2-7B-Chat-GPTQ`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Llama-2-7B-Chat-GPTQ] \
    --tokenizer=[path of Llama-2-7B-Chat-GPTQ] \
    --num-prompts 1 \
    --block-size=64 \
    --output-len=256 \
    --device=gcu \
    --dtype=float16 \
    --quantization=gptq \
    --gpu-memory-utilization=0.9 \
    --tensor-parallel-size=1
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model=[path of Llama-2-7B-Chat-GPTQ] \
    --tensor-parallel-size 1 \
    --max-model-len=4096 \
    --input-len=1024 \
    --output-len=1024 \
    --dtype=float16 \
    --device gcu \
    --num-prompts 1 \
    --block-size=64 \
    --quantization=gptq \
    --gpu-memory-utilization=0.9
```
注:
* 本模型支持的`max-model-len`为2048；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Llama-2-13B-chat-GPTQ

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [Llama-2-13B-chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ/tree/main)
* branch: `main`
* commit id: `ea078917a7e91c896787c73dba935f032ae658e9`

- 将上述url设定的路径下的内容全部下载到`Llama-2-13B-chat-GPTQ`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of Llama-2-13B-chat-GPTQ] \
    --tokenizer=[path of Llama-2-13B-chat-GPTQ] \
    --num-prompts 1 \
    --block-size=64 \
    --output-len=512 \
    --device=gcu \
    --dtype=float16 \
    --quantization=gptq \
    --gpu-memory-utilization=0.9 \
    --tensor-parallel-size=1
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model=[path of Llama-2-13B-chat-GPTQ] \
    --tensor-parallel-size 1 \
    --max-model-len=4096 \
    --input-len=512 \
    --output-len=512 \
    --dtype=float16 \
    --device gcu \
    --num-prompts 16 \
    --block-size=64 \
    --quantization=gptq \
    --gpu-memory-utilization=0.9
```
注:
* 本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama2-7b-w4a16c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* url: [Llama-2-7B-Chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)
* branch: `main`
* commit id: `d5ad9310836dd91b6ac6133e2e47f47394386cea`

- 将上述url设定的路径下的内容全部下载到`Llama-2-7B-Chat-w4a16c8`文件夹中。
- int8_kv_cache.json文件请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载，并拷贝到`Llama-2-7B-Chat-w4a16c8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
    --demo='te' \
    --model=[path of llama-2-7b-chat-gptq-W4A16C8] \
    --tokenizer=[path of llama-2-7b-chat-gptq-W4A16C8] \
    --output-len=128 \
    --device=gcu \
    --dtype=float16 \
    --quantization=gptq \
    --quantization-param-path=[path of int8_kv_cache.json] \
    --kv-cache-dtype=int8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model=[path of llama-2-7b-chat-gptq-W4A16C8] \
    --tensor-parallel-size 1 \
    --max-model-len=4096 \
    --input-len=1024 \
    --output-len=1024 \
    --dtype=float16 \
    --device gcu \
    --num-prompts 1 \
    --block-size=64 \
    --quantization=gptq \
    --quantization-param-path=[path of int8_kv_cache.json] \
    --kv-cache-dtype=int8 \
    --trust-remote-code
```
注:
* 本模型支持的`max-model-len`为4096；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


