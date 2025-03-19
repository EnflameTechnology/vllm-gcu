from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from vllm_utils.vision_language.models import resolve_vlm_input_cls
from vllm_utils.vision_language.datasets import resolve_dataset_cls
from vllm.transformers_utils.config import get_config
from vllm.outputs import RequestOutput
from vllm.config import HfOverrides

@dataclass
class MMRequestData:
    prompt: str
    vision_data: Dict
    answer: str = None
    outputs: Any = None
    sample_id: str = None
    origin_prompt: str = None


@dataclass
class MMRequest:
    stop_token_ids: Optional[List[int]]
    max_output_len: int
    mm_data_list: List[MMRequestData]


def get_model_arch(model, hf_overrides):
    hf_config = get_config(model, trust_remote_code=True)
    if hf_overrides:
        try:
            hf_config.update(hf_overrides)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse hf_overrides: {e}")
    model_arch = getattr(hf_config, "architectures", [""])[0]
    return model_arch

def get_vlm_input_obj(model: str,
                      model_arch_suffix: str,
                      tokenizer: Optional[str],
                      hf_overrides: Optional[HfOverrides] = None):
    model_arch = get_model_arch(model, hf_overrides) + model_arch_suffix
    vlm_input_obj = resolve_vlm_input_cls(model_arch)(model, tokenizer)
    return vlm_input_obj

def get_demo_mm_request(vlm_input_obj: Any,
                        question: str,
                        demo_vision_file: str,
                        max_output_len: int,
                        **input_kwargs):
    prompt = vlm_input_obj.get_demo_prompt(question, **input_kwargs)

    stop_token_ids = vlm_input_obj.get_stop_token_ids()

    vision_data = vlm_input_obj.get_demo_vision_data(
        demo_vision_file, **input_kwargs)

    return MMRequest(stop_token_ids=stop_token_ids,
                   max_output_len=max_output_len,
                   mm_data_list=[MMRequestData(
                       origin_prompt=question,
                       prompt=prompt,
                       vision_data=vision_data)])


def get_dummy_mm_request(vlm_input_obj: Any,
                         batch_size: int,
                         input_len: int,
                         input_vision_shape: str,
                         max_output_len: int,
                         **input_kwargs):
    image_feature_sizes = vlm_input_obj.get_image_feature_size(
        input_vision_shape, **input_kwargs)
    sum_image_feature_size = image_feature_sizes \
        if isinstance(image_feature_sizes,int) else sum(image_feature_sizes)
    if input_len < sum_image_feature_size:
        raise ValueError(
            f"input_len {input_len} less than sum of image feature size {sum_image_feature_size}")

    placeholder = vlm_input_obj.get_placeholder()

    stop_token_ids = vlm_input_obj.get_stop_token_ids()

    vision_data = vlm_input_obj.get_dummy_vision_data(
        input_vision_shape, **input_kwargs)
    
    input_kwargs.update({"input_vision_shape":input_vision_shape})

    prompt = vlm_input_obj.get_dummy_prompt(
                input_len, placeholder, image_feature_sizes, **input_kwargs)

    return MMRequest(stop_token_ids=stop_token_ids,
                   max_output_len=max_output_len,
                   mm_data_list=[MMRequestData(
                       prompt=prompt,
                       vision_data=vision_data)] * batch_size)


def get_dataset_mm_request(vlm_input_obj: Any,
                           dataset_name: str,
                           dataset_file: str,
                           max_output_len: int,
                           num_prompts: int = -1,
                           **input_kwargs):
    pre_process_func = vlm_input_obj.dataset_pre_process
    modality = vlm_input_obj.get_modality()

    datasets_obj = resolve_dataset_cls(dataset_name)()
    dataset = datasets_obj.build(
        dataset_file,
        pre_process_func,
        modality,
        num_prompts,
        **input_kwargs)

    stop_token_ids = vlm_input_obj.get_stop_token_ids()

    return MMRequest(stop_token_ids=stop_token_ids,
                   max_output_len=max_output_len,
                   mm_data_list=dataset)


def dataset_evaluate(vlm_input_obj: Any,
                     dataset_name: str,
                     inputs: MMRequestData,
                     outputs: List[RequestOutput]):
    post_process_func = vlm_input_obj.dataset_pred_post_process

    datasets_obj = resolve_dataset_cls(dataset_name)()
    acc = datasets_obj.evaluate(inputs, outputs, post_process_func,
                                vlm_input_obj.tokenizer)
    return acc
