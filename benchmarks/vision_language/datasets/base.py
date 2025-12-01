from abc import ABC, abstractmethod


class BaseDataset:
    @abstractmethod
    def build(self, dataset_file, pre_process_func, modality, num_prompts, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, inputs, outputs, dataset_pred_post_process, tokenizer):
        raise NotImplementedError
