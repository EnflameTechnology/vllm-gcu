from doctest import OutputChecker
# import osp
from vllm_utils.vision_language.interface import MMRequestData
from vllm_utils.vision_language.datasets.base import BaseDataset
from vllm_utils.vision_language.datasets.videomme_utils.videomme import VideoMME
import numpy as np
from vllm_utils.vision_language.datasets.videomme_utils.smp import *


class VideoMMEDataset:
    dataset = None

    def __init__(self):
        self.pack = False

    def build(self, dataset_file, pre_process_func, modality, num_prompts,
              **kwargs):
        global videomme_dataset_file
        videomme_dataset_file = dataset_file
        dataset = VideoMME('Video-MME',
                           use_subtitle=False,
                           dataset_path=dataset_file)
        num_frame = kwargs["num_frames"] if kwargs["num_frames"] else 8
        res = {}
        dataset_name = dataset.dataset_name

        sample_indices = list(dataset.videosp) if self.pack else list(
            dataset.data['index'])
        samples = list(dataset.videosp) if self.pack else list(
            range(len(dataset.data)))
        sample_map = {i: s for i, s in zip(sample_indices, samples)}

        sample_indices_sub = sample_indices
        if np.all([idx in res for idx in sample_indices_sub]):
            return model_name
        sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

        vllm_samples = []

        answers = dataset.data.values
        for i, idx in tqdm(enumerate(sample_indices_subrem)):
            if idx in res:
                continue
            # adapt to model frame sample number first
            # nframe = getattr(model, 'nframe', 0) if getattr(model, 'nframe', 0) > 0 else nframe

            # when using video-llm, build prompt returns video+question; otherwise, several frames+question
            struct = dataset.build_prompt(sample_map[idx],
                                          num_frames=num_frame,
                                          video_llm=False)
            prompts, vision_data = pre_process_func(dataset_name, struct, **kwargs)
            ans = answers[idx][-1]
            vllm_samples.append(
                MMRequestData(prompt=prompts,
                              answer=ans,
                              sample_id=idx,
                              vision_data=vision_data))

        return vllm_samples

    def evaluate(self, inputs, outputs, dataset_pred_pospt_process, tokenizer):
        from vllm_utils.vision_language.datasets.videomme_utils.videomme_utils import get_dimension_rating, extract_characters_regex
        dataset = VideoMME('Video-MME',
                           use_subtitle=False,
                           dataset_path=videomme_dataset_file)
        data = dataset.data

        scores = []
        for idx in range(len(outputs)):
            pred = outputs[idx].outputs[0].text
            ans = inputs.mm_data_list[idx].answer

            if extract_characters_regex(pred) == '':
                data.loc[idx, 'score'] = -1
                score = -1
            else:
                data.loc[idx,
                         'score'] = int(extract_characters_regex(pred) == ans)
                score = int(extract_characters_regex(pred) == ans)

            scores.append(score)

        rating = get_dimension_rating(data)
        return rating
