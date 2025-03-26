# 大语言模型数据集精度验证

本节给出使用vLLM-gcu及opencompass完成大语言模型数据集精度验证的方法。

## 环境配置

请安装opencompass 0.3.1和vLLM-gcu 0.6.1.post2版本，调整依赖项版本，确保两者正常工作。

注：opencompass 0.3.1版本需要使用python 3.10或以上版本运行

## 方法

OpenCompass中已经官方支持vllm backend。适配于vLLM-gcu，使用规则如下：

1. opencompass支持在特定路径下（相对于根目录："configs/models"）通过静态文件配置参数。当测试只使用单卡时，`run_cfg`的`num_gpus`应设为0；当测试使用多卡时，`run_cfg`的`num_gpus`应设为拟使用的卡的数量，与model_kwargs的`tensor_parallel_size`相同，其他模型参数配置方法和可支持的配置参数符合opencompass中配置vllm模型测试的规则。
例如，配置文件为`configs/models/vllm_ppl_model.py`，文件内容如下：

    ```python
    # 单卡测试的configs/models/vllm_ppl_model.py
    from opencompass.models import VLLM

    models = [
        dict(
            type=VLLM,
            abbr='vllm-ppl-model',
            path='/path/to/model/',
            max_out_len=1,
            max_seq_len=4096,
            batch_size=32,
            generation_kwargs=dict(temperature=0),
            run_cfg=dict(num_gpus=0, num_procs=1),
            model_kwargs=dict(device='gcu', gpu_memory_utilization=0.7)
        )
    ]

    # 多卡测试的configs/models/vllm_ppl_model.py
    from opencompass.models import VLLM

    models = [
        dict(
            type=VLLM,
            abbr='vllm-ppl-model',
            path='/path/to/model/',
            max_out_len=1,
            max_seq_len=4096,
            batch_size=32,
            generation_kwargs=dict(temperature=0),
            run_cfg=dict(num_gpus=2, num_procs=1),
            model_kwargs=dict(device='gcu', gpu_memory_utilization=0.7, tensor_parallel_size=2)
        )
    ]

    ```

2. 测试入口为`opencompass/run.py`，当使用多卡测试时，应设置环境变量`CUDA_VISIBLE_DEVICES`为任意设备序号，设备序号数量需等于拟使用的gcu卡的数量，执行无需使用任何CUDA设备，测试命令如：
    ```bash
    # 单卡测试命令
    python3 run.py \
    --models=vllm_ppl_model \
    --datasets=mmlu_ppl

    # 多卡测试命令
    export CUDA_VISIBLE_DEVICES=0,1
    python3 run.py \
    --models=vllm_ppl_model \
    --datasets=mmlu_ppl
    ```
