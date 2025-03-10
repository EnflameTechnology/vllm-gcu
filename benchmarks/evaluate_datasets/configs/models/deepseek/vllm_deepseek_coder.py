from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='deepseek-coder-vllm',
        path='/home/pretrained_models/deepseek-coder-6.7b-base/',
        # prompt_replace_pattern=_prompt_replace_pattern,
        # generated_replace_pattern=_generate_replace_pattern,
        max_out_len=1024,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        # end_str='</s>',
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', max_model_len=4096, trust_remote_code=True)
    )
]

