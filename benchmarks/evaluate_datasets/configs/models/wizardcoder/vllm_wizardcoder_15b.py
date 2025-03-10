from opencompass.models import VLLM

_prompt_replace_pattern = dict(
    pattern=[
        dict(source="    ", target="\t"),
    ],
)

_generate_replace_pattern = dict(
    pattern=[
        dict(source=".*### Response:", target=""),
        dict(source="\t", target="    "),
        dict(source="\r", target=""),
        dict(source=".*problem:\n\n```python\n", target="", count=1),
        dict(source="\n```.*", target=""),
        dict(source='if __name__ == "__main__":.*', target='if __name__ == "__main__":'),
        dict(source="# Example usage.*", target=""),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='WizardCoder-15B-V1.0-vllm',
        path='/home/pretrained_models/WizardCoder-15B-v1.0/',
        # prompt_replace_pattern=_prompt_replace_pattern,
        # generated_replace_pattern=_generate_replace_pattern,
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        generation_kwargs=dict(temperature=0),
        # end_str='</s>',
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', trust_remote_code=True)
    )
]
