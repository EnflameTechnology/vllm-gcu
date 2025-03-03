import os
from vllm_utils.vision_language.interface import MMRequestData
from vllm_utils.vision_language.datasets.base import BaseDataset


class LlavaBenchCocoDataset(BaseDataset):
    def __init__(self) -> None:
        self.dataset_name = "llava-bench-coco"

    def build(self, dataset_file,
              pre_process_func, modality, num_prompts: int,
              **kwargs):
        from datasets import load_dataset
        dataset = load_dataset(
            dataset_file.split(".")[-1], data_files=dataset_file)['train']

        llava_bench_coco = []
        for index, data in enumerate(dataset):
            if num_prompts > 0 and index >= num_prompts:
                break

            prompt = data["question"]
            prompt = pre_process_func(self.dataset_name, prompt, **kwargs)
            image = data["image"].convert("RGB")
            llava_bench_coco.append(MMRequestData(
                prompt=prompt,
                vision_data={modality: image},
                answer=data["answer"]
            ))

        return llava_bench_coco

    def evaluate(self, inputs, outputs, dataset_pred_post_process, tokenizer):
        import evaluate
        import numpy as np

        golden = []
        for input in inputs.mm_data_list:
            golden.append(input.answer)

        pred = []
        for output in outputs:
            pred.append(output.outputs[0].text)

        ROOT_DIR = os.path.dirname(__file__)
        rouge_path = os.path.join(ROOT_DIR, "rouge.py")
        if os.path.exists(rouge_path):
            metric = evaluate.load(rouge_path)
        else:
            metric = evaluate.load("rouge")
        rouges = metric.compute(
            predictions=pred,
            references=golden,
            use_stemmer=True,
            use_aggregator=False,
            tokenizer=tokenizer.tokenize,
        )

        return {k: round(np.mean(v) * 100, 4) for k, v in rouges.items()}
