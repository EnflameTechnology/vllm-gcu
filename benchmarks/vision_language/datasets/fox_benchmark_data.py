import os
import json

from PIL import Image

from vllm_utils.vision_language.interface import MMRequestData
from vllm_utils.vision_language.datasets.base import BaseDataset


class FoxBenchmarkData(BaseDataset):
    def __init__(self):
        self.dataset_name = "fox_benchmark_data"
        self.default_ocr_dir = "cn_pdf_png"
        self.default_ocr_answer = "cn_page_ocr.json"

    def build(
        self, dataset_file, pre_process_func, modality, num_prompts, **kwargs
    ):
        image_dir = os.path.join(dataset_file, self.default_ocr_dir)
        answer_file = os.path.join(dataset_file, self.default_ocr_answer)
        with open(answer_file, "r", encoding="UTF-8") as f:
            answer_list = json.load(f)

        fox_benchmark_list = []
        for index, data in enumerate(answer_list):
            if num_prompts > 0 and index >= num_prompts:
                break

            prompt = data["conversations"][0]["value"]
            prompt = pre_process_func(self.dataset_name, prompt, **kwargs)
            image_path = os.path.join(image_dir, data["image"])
            image = Image.open(image_path).convert("RGB")
            answer = data["conversations"][1]["value"]
            fox_benchmark_list.append(
                MMRequestData(
                    prompt=prompt, vision_data={modality: image}, answer=answer
                )
            )


        return fox_benchmark_list

    def evaluate(self, inputs, outputs, dataset_pred_post_process, tokenizer):
        from vllm_utils.vision_language.datasets.fox_benchmark_data_utils.eval import (
            doc_text_eval,
        )

        results_list = []
        for index, input in enumerate(inputs.mm_data_list):
            output = outputs[index]
            preditc_text = output.outputs[0].text
            preditc_text = dataset_pred_post_process(
                self.dataset_name, preditc_text
            )
            results_list.append(
                {"predict": preditc_text, "answer": input.answer}
            )

        metrict_dict = doc_text_eval(results_list)

        return metrict_dict
