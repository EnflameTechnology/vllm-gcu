## llama

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

### llama2-70b

本模型推理及性能测试需要四张enflame gcu。

#### 模型下载
*  url:[llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-hf)

*  branch:`main`

*  commit id:`6aa89cf`

将上述url设定的路径下的内容全部下载到`llama-2-70b-hf`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama-2-70b-hf] \
 --tensor-parallel-size=8 \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --gpu-memory-utilization=0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama-2-70b-hf] \
 --tensor-parallel-size=4 \
 --max-model-len=4096 \
 --tokenizer=[path of llama-2-70b-hf] \
 --input-len=512 \
 --output-len=240 \
 --num-prompts=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --dtype=float16
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

将下面的配置信息存为一个python文件，放入OpenCompass中如下路径`configs/models/llama/vllm_llama2_70b.py`

```python
from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='llama2-70b-vllm',
        path='/path/to/Llama2-70b',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu',
                          tensor_parallel_size=4,
                          enforce_eager=True)
    )
]

```

执行以下命令

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 run.py \
 --models=vllm_llama2_70b \
 --datasets=mmlu_gen
```

### Meta-Llama-3-8B
#### 模型下载
*  url:[Meta-Llama-3-8B](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B/files)

*  branch:`master`

*  commit id:`e4260355`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3-8B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3-8B] \
 --demo=te \
 --dtype=float16 \
 --output-len=20 \
 --max-model-len=64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3-8B] \
 --max-model-len=8192 \
 --tokenizer=[path of Meta-Llama-3-8B] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### Meta-Llama-3-70B
#### 模型下载
*  url:[Meta-Llama-3-70B](https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-70B/files)

*  branch:`master`

*  commit id:`0061f2a0`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3-70B`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3-70B] \
 --tensor-parallel-size=4 \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --max-model-len=4096
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3-70B] \
 --tensor-parallel-size=4 \
 --max-model-len=8192 \
 --tokenizer=[path of Meta-Llama-3-70B] \
 --input-len=1024 \
 --output-len=7168 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


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


### llama2-13b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2-13b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`llama2-13b-w8a16_gptq`文件夹中。
- `llama2-13b-w8a16_gptq`目录结构如下所示：

```shell
llama2-13b-w8a16_gptq/
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
 --model=[path of llama2-13b-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --quantization gptq \
 --max-model-len 64
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama2-13b-w8a16_gptq] \
 --max-model-len=4096 \
 --tokenizer=[path of llama2-13b-w8a16_gptq] \
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

### llama2-70b-w8a16_gptq

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2-70b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`llama2-70b-w8a16_gptq`文件夹中。
- `llama2-70b-w8a16_gptq`目录结构如下所示：

```shell
llama2-70b-w8a16_gptq/
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
 --model=[path of llama2-70b-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=20 \
 --quantization gptq \
 --tensor-parallel-size=2
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama2-70b-w8a16_gptq] \
 --tensor-parallel-size=2 \
 --max-model-len=4096 \
 --tokenizer=[path of llama2-70b-w8a16_gptq] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama3-8b-w8a16_gptq

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama3-8b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`llama3-8b-w8a16_gptq`文件夹中。
- `llama3-8b-w8a16_gptq`目录结构如下所示：

```shell
llama3-8b-w8a16_gptq/
├── config.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama3-8b-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --quantization gptq \
 --max-model-len 64 \
 --output-len 20
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama3-8b-w8a16_gptq] \
 --max-model-len=8192 \
 --tokenizer=[path of llama3-8b-w8a16_gptq] \
 --input-len=128 \
 --output-len=3968 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq \
 --enforce-eager
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### llama3-70b-w8a16_gptq

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama3-70b-w8a16_gptq.tar`文件并解压，将压缩包内的内容全部拷贝到`llama3-70b-w8a16_gptq`文件夹中。
- `llama3-70b-w8a16_gptq`目录结构如下所示：

```shell
llama3-70b-w8a16_gptq/
├── config.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
└── tokenizer.model
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama3-70b-w8a16_gptq] \
 --demo=te \
 --dtype=float16 \
 --output-len=256 \
 --quantization gptq \
 --tensor-parallel-size=2 \
 --gpu-memory-utilization=0.945 \
 --max-model-len=4096
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama3-70b-w8a16_gptq] \
 --tensor-parallel-size=2 \
 --max-model-len=8192 \
 --tokenizer=[path of llama3-70b-w8a16_gptq] \
 --input-len=1024 \
 --output-len=7168 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --quantization gptq \
 --gpu-memory-utilization=0.945
```
注：
*  本模型支持的`max-model-len`为8192；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Meta-Llama-3.1-8B-Instruct

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/tree/main/)

*  branch: `main`

*  commit id: `8c22764`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-8B-Instruct`文件夹中。

#### requirements

```shell
python3 -m pip install transformers==4.48.2
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3.1-8B-Instruct] \
 --demo=te \
 --dtype=bfloat16 \
 --output-len=256 \
 --device=gcu \
 --max-model-len=32768 \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-8B-Instruct] \
 --max-model-len=32768 \
 --tokenizer=[path of Meta-Llama-3.1-8B-Instruct] \
 --input-len=8192 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=bfloat16 \
 --device gcu \
 --tensor-parallel-size 1 \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为131072, 单张卡可跑32768；

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


### Meta-Llama-3.1-70B-Instruct

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [Meta-Llama-3.1-70B-Instruct](https://modelscope.cn/models/llm-research/meta-llama-3.1-70b-instruct/files)

*  branch: `master`

*  commit id: `b6444261`

将上述url设定的路径下的内容全部下载到`Meta-Llama-3.1-70B-Instruct`文件夹中。

#### requirements

```shell
python3 -m pip install transformers==4.48.2
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3.1-70B-Instruct] \
 --tensor-parallel-size=8 \
 --demo=te \
 --max-model-len=32768 \
 --dtype=bfloat16 \
 --device=gcu \
 --output-len=256 \
 --gpu-memory-utilization 0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-70B-Instruct] \
 --max-model-len=32768 \
 --tokenizer=[path of Meta-Llama-3.1-70B-Instruct] \
 --tensor-parallel-size=8 \
 --input-len=8192 \
 --output-len=512 \
 --num-prompts=1 \
 --block-size=64 \
 --device=gcu \
 --dtype=bfloat16 \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为131072, 需8张卡跑32768；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### llama3-70b-w4a16

本模型推理及性能测试需要2张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Meta-Llama-3-70B_W4A16_GPTQ.tar`文件以及并解压，将压缩包内的内容全部拷贝到`llama3-70b-w4a16`文件夹中。
- `llama3-70b-w4a16`目录结构如下所示：

```shell
llama3-70b-w4a16/
  ├── config.json
  ├── model.safetensors
  ├── quantize_config.json
  ├── tokenizer_config.json
  ├── tokenizer.json
  └── tops_quantize_info.json
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama3-70b-w4a16] \
 --tensor-parallel-size=2 \
 --max-model-len=8192 \
 --output-len=512 \
 --demo=te \
 --dtype=float16 \
 --quantization=gptq \
 --device=gcu
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama3-70b-w4a16] \
 --device=gcu \
 --max-model-len=8192 \
 --tokenizer=[path of llama3-70b-w4a16] \
 --input-len=2048 \
 --output-len=1024 \
 --num-prompts=1 \
 --tensor-parallel-size=2 \
 --block-size=64
```
注:
* 本模型支持的`max-model-len`8192;
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



### llama2-70b-w4a16c8

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2_70b_w4a16c8.tar`文件以及并解压，将压缩包内的内容全部拷贝到`llama2_70b_w4a16c8.tar`文件夹中。
- `llama2_70b_w4a16c8.tar`目录结构如下所示：

```shell
.
├── config.json
├── int8_kv_cache.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json
```

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
    --kv-cache-dtype=int8 \
    --tensor-parallel-size 4
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test \
    --perf \
    --model=[path of llama-2-7b-chat-gptq-W4A16C8] \
    --tensor-parallel-size 4 \
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

### llama2-70b-w8a8c8

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2-70b-w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`llama2-70b-w8a8`文件夹中。
- int8_kv_cache.json文件请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载，并拷贝到`llama2-70b-w8a8`文件夹中。
- `llama2-70b-w8a8`目录结构如下所示：

```shell
llama2-70b-w8a8/
├── int8_kv_cache.json
├── tops_quantize_info.json
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
 --model=[path of llama2-70b-w8a8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --tensor-parallel-size 4 \
 --max-model-len=2048 \
 --demo=te \
 --dtype=float16 \
 --output-len=128 \
 --device gcu \
 --kv-cache-dtype int8 \
 --gpu-memory-utilization 0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama2-70b-w8a8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --tensor-parallel-size 4 \
 --max-model-len=2048 \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --kv-cache-dtype int8 \
 --device gcu \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Meta-Llama-3.1-70B-Instruct-w4a16

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载

* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Meta-Llama-3.1-70B-Instruct_W4A16_AWQ.tar`文件以及并解压，将压缩包内的内容全部拷贝到`Meta-Llama-3.1-70B-Instruct_W4A16_AWQ`文件夹中。
- `Meta-Llama-3.1-70B-Instruct_W4A16_AWQ`目录结构如下所示：

```shell
Meta-Llama-3.1-70B-Instruct_W4A16_AWQ/
├── config.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### requirements

```shell
python3 -m pip install transformers==4.48.2
```

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Meta-Llama-3.1-70B-Instruct_W4A16_AWQ] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --dtype=float16 \
 --device=gcu \
 --output-len=256 \
 --demo=te \
 --gpu-memory-utilization=0.9 \
 --quantization=awq
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-70B-Instruct_W4A16_AWQ] \
 --tokenizer=[path of Meta-Llama-3.1-70B-Instruct_W4A16_AWQ] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --dtype=float16 \
 --device=gcu \
 --input-len=31744 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --gpu-memory-utilization=0.8 \
 --quantization=awq
```
注:
* 本模型支持的`max-model-len`为131072，4张S60可跑32768；
*  `input-len`、`output-len`和`num-prompts`可按需调整；
* 配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;


### llama2_7b_chat_w8a8c8

本模型推理及性能测试需要1张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`llama2_7b_chat_w8a8.tar`文件并解压，将压缩包内的内容全部拷贝到`llama2_7b_chat_w8a8`文件夹中。
- int8_kv_cache_chat.json文件请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载，并拷贝到`llama2_7b_chat_w8a8`文件夹中。
- `llama2_7b_chat_w8a8`目录结构如下所示：

```shell
llama2_7b_chat_w8a8
├── config.json
├── configuration.json
├── generation_config.json
├── int8_kv_cache_chat.json
├── model.safetensors
├── quantize_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── tokenizer.model
└── tops_quantize_info.json
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of llama2_7b_chat_w8a8] \
 --quantization-param-path [path of int8_kv_cache_chat.json] \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --demo=te \
 --dtype=float16 \
 --output-len=128 \
 --device gcu \
 --kv-cache-dtype int8 \
 --gpu-memory-utilization 0.9
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of llama2_7b_chat_w8a8] \
 --quantization-param-path [path of int8_kv_cache_chat.json] \
 --tensor-parallel-size 1 \
 --max-model-len=4096 \
 --input-len=1024 \
 --output-len=1024 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --kv-cache-dtype int8 \
 --device gcu \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Meta-Llama-3.1-70B-Instruct_W8A8C8

本模型推理及性能测试需要4张enflame gcu。

#### 模型下载
* 如需要下载权重，请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载

- 下载`Meta-Llama-3.1-70B-Instruct_W8A8.tar`文件并解压，将压缩包内的内容全部拷贝到`Meta-Llama-3.1-70B-Instruct_W8A8`文件夹中。
- int8_kv_cache.json文件请联系商务人员开通[EGC](https://egc.enflame-tech.com/)权限进行下载，并拷贝到`Meta-Llama-3.1-70B-Instruct_W8A8`文件夹中。
- `Meta-Llama-3.1-70B-Instruct_W8A8`目录结构如下所示：

```shell
Meta-Llama-3.1-70B-Instruct_W8A8
├── config.json
├── int8_kv_cache.json
├── model.safetensors
├── quantize_config.json
├── tokenizer_config.json
├── tokenizer.json
└── tops_quantize_info.json
```

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --demo=te \
 --model=[path of Meta-Llama-3.1-70B-Instruct_W8A8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --kv-cache-dtype int8 \
 --quantization=w8a8 \
 --max-model-len=2048 \
 --tensor-parallel-size 4 \
 --dtype=float16 \
 --output-len=128 \
 --device gcu \
 --gpu-memory-utilization 0.8
```

#### 性能测试

```shell
python3 -m vllm_utils.benchmark_test --perf \
 --model=[path of Meta-Llama-3.1-70B-Instruct_W8A8] \
 --quantization-param-path [path of int8_kv_cache.json] \
 --kv-cache-dtype int8 \
 --tensor-parallel-size 4 \
 --max-model-len=2048 \
 --input-len=512 \
 --output-len=128 \
 --num-prompts=1 \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --gpu-memory-utilization 0.9
```
注：
*  本模型支持的`max-model-len`为4096；

*  `input-len`、`output-len`和`num-prompts`可按需调整；

*  配置 `output-len`为1时,输出内容中的`latency`即为time_to_first_token_latency;

### Llama-3.3-70B-Instruct

本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [Llama-3.3-70B-Instruct](https://www.modelscope.cn/models/LLM-Research/Llama-3.3-70B-Instruct/files)

*  branch: `master`

*  commit id: `a5b145fa`

将上述url设定的路径下的内容全部下载到`Llama-3.3-70B-Instruct`文件夹中。

#### 批量离线推理

```shell
python3 -m vllm_utils.benchmark_test \
 --model=[path of Llama-3.3-70B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=8 \
 --output-len=256 \
 --demo=te \
 --gpu-memory-utilization=0.9 \
 --device=gcu
```

#### serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model=[path of Llama-3.3-70B-Instruct] \
 --tokenizer=[path of Llama-3.3-70B-Instruct] \
 --dtype=bfloat16 \
 --max-model-len=32768 \
 --tensor-parallel-size=8 \
 --block-size=64 \
 --gpu-memory-utilization=0.9 \
 --disable-log-stats \
 --device=gcu

# 启动客户端
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --model=[path of Llama-3.3-70B-Instruct] \
 --tokenizer=[path of Llama-3.3-70B-Instruct] \
 --request-rate=inf \
 --random-input-len=1024 \
 --random-output-len=1024 \
 --num-prompts=1 \
 --dataset-name=random \
 --ignore-eos \
 --strict-in-out-len
```
注：
* 本模型支持的`max-model-len`为131072，8张S60可跑32768；
* 为保证输入输出长度固定，数据集使用随机数测试；
* num-prompts, random-input-len和random-output-len可按需调整；