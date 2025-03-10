from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='llama-3-70b',
        path='/home/pretrained_models/llama-3-70b/',
        max_out_len=10,
        max_seq_len=8192,
        batch_size=1,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', max_model_len=8192, tensor_parallel_size=8, enforce_eager=False, trust_remote_code=True, quantization='gptq', dtype='float16')
    )
]

