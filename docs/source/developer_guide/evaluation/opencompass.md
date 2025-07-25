# Using OpenCompass 
This document will guide you have a accuracy testing using [OpenCompass](https://github.com/open-compass/opencompass).

## 1. Online Serving

You can run docker container to start the vLLM server on a single gcu:

```{code-block} bash
   :substitutions:
export VLLM_USE_MODELSCOPE=True
python3 -m vllm.entrypoints.openai.api_server \
 --model Qwen/Qwen2.5-7B-Instruct \
 --tensor-parallel-size 1 \
 --max-model-len 26240 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code
```

Once the server is started, you can query the model with given prompts:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "prompt": "The popular LLM models are",
        "max_tokens": 256,
        "temperature": 0
    }'
```

## 2. Run ceval accuracy test with OpenCompass

Install OpenCompass (for python3.10)

```bash
conda create -n opencompass python=3.10
conda activate opencompass
pip install opencompass modelscope[framework]
export DATASET_SOURCE=ModelScope
git clone https://github.com/open-compass/opencompass.git
```

Save the following demo code into `opencompass_gcu.py`

```python
from mmengine.config import read_base
from opencompass.models import OpenAISDK

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

datasets = ceval_datasets[:1] # ceval-computer_network dataset

api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', api_role='SYSTEM')],
)

models = [
    dict(
        abbr='Qwen2.5-7B-Instruct-vLLM-API',
        type=OpenAISDK,
        key='EMPTY',
        openai_api_base='http://127.0.0.1:8000/v1', 
        path='Qwen/Qwen2.5-7B-Instruct', 
        tokenizer_path='Qwen/Qwen2.5-7B-Instruct', 
        rpm_verbose=True, 
        meta_template=api_meta_template,
        query_per_second=1, 
        max_out_len=4096, 
        max_seq_len=26240, 
        temperature=0, 
        batch_size=4,
        retry=5,
    )
]
```

Run the evalution:

```
python3 run.py opencompass_gcu.py --debug
```

Sample outputs:

```
| dataset | version | metric | mode | Qwen2.5-7B-Instruct-vLLM-API |
|----- | ----- | ----- | ----- | -----|
| ceval-computer_network | db9ce2 | accuracy | gen | 69.31 |
```

More details can be found in [OpenCompass](https://opencompass.readthedocs.io/en/latest/index.html).
