import argparse
import dataclasses
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from vllm import RequestOutput
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm_utils.vision_language.datasets import VLM_DATASETS
from vllm_utils.vision_language.interface import (
    dataset_evaluate,
    get_dataset_mm_request,
    get_demo_mm_request,
    get_dummy_mm_request,
    get_vlm_input_obj,
)

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)


def run_vllm(
    requests,
    n: int,
    engine_args,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    frequency_penalty: float,
    repetition_penalty: float,
    batch_size: int = 1,
    num_iters: int = 1,
    ignore_eos: bool = True,
    **kwargs,
) -> Tuple[float, Optional[float], List["RequestOutput"]]:
    from vllm import LLM, SamplingParams

    llm_build_dict = dataclasses.asdict(engine_args)
    llm_build_dict.update(kwargs)
    llm = LLM(**llm_build_dict)

    dummy_sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        ignore_eos=True,
        max_tokens=1,
        stop_token_ids=requests.stop_token_ids,
    )
    llm.generate(
        prompts={
            "prompt": requests.mm_data_list[0].prompt,
            "multi_modal_data": requests.mm_data_list[0].vision_data,
        },
        sampling_params=dummy_sampling_params,
        use_tqdm=False,
    )

    def clear(metrics):
        for metric in vars(metrics):
            prometheus = getattr(metrics, metric)
            if hasattr(prometheus, "clear"):
                prometheus.clear()

    if not engine_args.disable_log_stats:
        clear(llm.llm_engine.stat_loggers["prometheus"].metrics)

    start = time.perf_counter()
    num_request = len(requests.mm_data_list)
    outputs_result = []
    if batch_size > num_request:
        logging.warning("cast batch_size to length of requests!")
        batch_size = num_request
    for _ in range(num_iters):
        # Add the requests to the engine.
        for j in tqdm(range(math.ceil(num_request / batch_size))):
            tmp_req = requests.mm_data_list[j * batch_size : (j + 1) * batch_size]
            inputs = []
            sampling_params_list = []
            for item in tmp_req:
                inputs.append(
                    {
                        "prompt": item.prompt,
                        "multi_modal_data": item.vision_data,
                    }
                )
                sampling_params = SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    repetition_penalty=repetition_penalty,
                    ignore_eos=ignore_eos,
                    max_tokens=requests.max_output_len,
                    stop_token_ids=requests.stop_token_ids,
                )
                sampling_params_list.append(sampling_params)

            outputs = llm.generate(
                inputs, sampling_params=sampling_params_list, use_tqdm=True
            )
            outputs_result.extend(outputs)
    end = time.perf_counter()
    print(f"run {num_iters} cost {end-start}")
    if not engine_args.disable_log_stats:
        time_per_output_token = llm.llm_engine.stat_loggers[
            "prometheus"
        ].metrics.histogram_time_per_output_token
        samples = time_per_output_token.collect()[0].samples
        decode_elapsed_time = (
            samples[-1].value / num_iters if len(samples) > 0 else None
        )
    else:
        # disable_log_stats
        decode_elapsed_time = None
    return (
        (end - start) / num_iters,
        decode_elapsed_time,
        outputs_result,
    )


def vllm_perf(
    requests: List[Tuple[str, str, int, int]],
    outputs: List["RequestOutput"],
    elapsed_time: float,
    decode_elapsed_time: Optional[float],
) -> Dict[str, str]:
    # perf info
    prefill_times = []
    for output in outputs:
        if output.metrics.first_token_time is None:
            print("Warning: None found in first_token_time.")
        else:
            prefill_times.append(
                output.metrics.first_token_time - output.metrics.first_scheduled_time
            )
    prefill_latency_per_token = np.mean(prefill_times)

    output_len = [len(item.outputs[0].token_ids) for item in outputs]
    max_output_len = max(output_len)
    total_output_tokens = sum(output_len)
    total_decode_tokens = total_output_tokens - len(requests.mm_data_list)
    if decode_elapsed_time is None:
        print("decode elapsed time is not recorded as set `--disable_log_stats`")
        decode_latency_per_token = None
        decode_throughput = None
    else:
        decode_latency_per_token = (
            f"{decode_elapsed_time / total_decode_tokens * 1000:.2f} ms"
        )
        decode_throughput = f"{total_decode_tokens / decode_elapsed_time :.2f} tokens/s"

    perf_info = dict(
        total_output_tokens=total_output_tokens,
        latency_num_prompts=f"{elapsed_time * 1000:.2f} ms",
        latency_per_token=f"{elapsed_time * 1000 / max_output_len:.2f} ms",
        request_per_second=f"{len(requests.mm_data_list) / elapsed_time:.2f} requests/s",
        token_per_second=f"{total_output_tokens / elapsed_time:.2f} tokens/s",
        prefill_latency_per_token=f"{prefill_latency_per_token * 1000:.2f} ms",
        decode_latency_per_token=decode_latency_per_token,
        decode_throughput=decode_throughput,
    )

    return perf_info


def save_to_csv(csv_file, data):
    import csv

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        writer.writerows(data.items())

    return csv_file


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    msgs = vars(args)

    from vllm.transformers_utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)

    vlm_input_obj = get_vlm_input_obj(
        args.model, args.model_arch_suffix, tokenizer, args.hf_overrides)

    input_kwargs = {}
    if vlm_input_obj.modality == "video":
        input_kwargs.update({"num_frames": args.num_frames})

    input_kwargs.update({"mm_per_prompt": args.mm_per_prompt})

    ignore_eos = False
    if args.demo:
        requests = get_demo_mm_request(
            vlm_input_obj,
            args.prompt,
            args.input_vision_file,
            args.max_output_len,
            **input_kwargs,
        )
    elif args.acc:
        requests = get_dataset_mm_request(
            vlm_input_obj,
            args.dataset_name,
            args.dataset_file,
            args.max_output_len,
            args.num_prompts,
            **input_kwargs,
        )
    elif args.perf:
        ignore_eos = True
        requests = get_dummy_mm_request(
            vlm_input_obj,
            args.batch_size,
            args.input_len,
            args.input_vision_shape,
            args.max_output_len,
            **input_kwargs,
        )
    else:
        raise ValueError("Unsupported mode.")

    kwargs = {}
    if args.max_num_seqs:
        kwargs["max_num_seqs"] = args.max_num_seqs
    if args.mm_per_prompt > 1:
        kwargs.update({"limit_mm_per_prompt": {"image": args.mm_per_prompt}})

    if args.backend == "vllm":
        elapsed_time, decode_elapsed_time, outputs = run_vllm(
            requests,
            args.n,
            EngineArgs.from_cli_args(args),
            args.temperature,
            args.top_p,
            args.presence_penalty,
            args.frequency_penalty,
            args.repetition_penalty,
            args.batch_size,
            args.num_iters,
            ignore_eos,
            **kwargs,
        )

        if args.demo:
            for request, output in zip(requests.mm_data_list, outputs):
                print(
                    f"Prompt: {request.origin_prompt}, "
                    f"Generated text: {output.outputs[0].text}"
                )

        if args.perf:
            perf_info = vllm_perf(requests, outputs, elapsed_time, decode_elapsed_time)
            msgs.update(perf_info)
            print("\n***Perf Info***")
            print(json.dumps(perf_info, indent=4))

        if args.acc:
            acc_info = dataset_evaluate(
                vlm_input_obj, args.dataset_name, requests, outputs
            )
            msgs.update(acc_info)
            print("\n***Acc Info***")
            print(json.dumps(acc_info, indent=4))
            print(f"elapsed time:{elapsed_time * 1000:.2f} ms")

        if args.save_output:
            infer_result = []
            for request, output in zip(requests.mm_data_list, outputs):
                infer_result.append(
                    {
                        "prompt": request.origin_prompt,
                        "answer": request.answer,
                        "result": output.outputs[0].text,
                    }
                )
            with open(args.save_output, "w", encoding="UTF-8") as f:
                json.dump(infer_result, f, indent=4)
        csvfile = save_to_csv(f"{time.strftime('%Y%m%d%H%M%S')}.csv", msgs)
        print(f"save to {csvfile}")
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    total_num_tokens = sum(
        len(item.prompt_token_ids) + len(item.outputs[0].token_ids) for item in outputs
    )
    print(
        f"Throughput: {len(requests.mm_data_list) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark vision language model.")
    parser.add_argument("--backend", type=str, choices=["vllm"], default="vllm")

    parser.add_argument("--model-arch-suffix", type=str, default="")
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument("--num-iters", type=int, default=1, help="Number of iteration.")

    # offline inference args
    parser.add_argument("--demo", action="store_true", help="offline inference example")
    parser.add_argument(
        "--prompt",
        type=str,
        default="\nUSER: What is the content of this image?\nASSISTANT:",
    )
    parser.add_argument("--input-vision-file", type=str, default=None)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from the video.",
    )

    # performance args
    parser.add_argument("--perf", action="store_true", help="readout perf")
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=128,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument("--input-vision-shape", type=str, default=None)

    # dataset test args
    parser.add_argument("--acc", action="store_true", help="acc on dataset")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        choices=VLM_DATASETS.keys(),
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--dataset-file", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=-1,
        help="Number of prompts to process,only used in dataset test.",
    )
    parser.add_argument("--batch-size", type=int, default=1)

    # inference args
    parser.add_argument("--save-output", type=str, default=None)
    parser.add_argument(
        "--mm-per-prompt",
        type=int,
        default=1,
        help="number of multi modal file per prompt",
    )

    # sampling args
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on whether they appear in the generated text so far. default=0.0 (no penalty)",
    )
    parser.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Float that penalizes new tokens based on their frequency in the generated text so far. default=0.0 (no penalty)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. default=1.0 (no penalty)",
    )

    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
