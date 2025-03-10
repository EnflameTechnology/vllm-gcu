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
        dict(source=".*```python\n", target="", count=1),
        dict(source="\n```.*", target=""),
        dict(source='if __name__ == "__main__":.*', target='if __name__ == "__main__":'),
        dict(source="# Example usage.*", target=""),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr='WizardCoder-33B-V1.1-vllm',
        path='/home/pretrained_models/WizardCoder-33B-v1.1/',
        # prompt_replace_pattern=_prompt_replace_pattern,
        # generated_replace_pattern=_generate_replace_pattern,
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=8,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=0, num_procs=1),
        model_kwargs=dict(device='gcu', gpu_memory_utilization=0.85, dtype="half", tensor_parallel_size=2, max_model_len=4096)
    )
]
