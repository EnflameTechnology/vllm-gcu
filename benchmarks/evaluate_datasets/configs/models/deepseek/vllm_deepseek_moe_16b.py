from opencompass.models import VLLM

models = [
    dict(
        type=VLLM,
        abbr='deepseek-moe-vllm',
        path='/home/pretrained_models/deepseek-moe-16b-chat/',
        # prompt_replace_pattern=_prompt_replace_pattern,
        # generated_replace_pattern=_generate_replace_pattern,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0),
        # end_str='</s>',
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', dtype='float16', trust_remote_code=True)
    )
]

