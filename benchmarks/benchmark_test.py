import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import torch
import uvloop
import vllm  # noqa
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLMEngine, PoolingRequestOutput, RequestOutput, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.utils import Counter, FlexibleArgumentParser, merge_async_iterators
from vllm.v1.engine.llm_engine import LLMEngine as V1Engine

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)


TASK_MAP = {
    "te": "text-english",
    "tc": "text-chinese",
    "ch": "chat",
    "chc": "character-chat",
    "cc": "code-completion",
    "ci": "code-infilling",
    "cin": "code-instruction",
    "dch": "deepseek-chat",
    "cm": "code-merge",
}


def read_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    chat_template: Optional[str] = None,
    add_generation_prompt: Optional[bool] = False,
):
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    template_original_prompts = None
    orig_prompts = [prompt for prompt, _ in dataset]
    if chat_template is not None:
        if chat_template != "default":
            with open(chat_template, "r") as f:
                template = f.read()
            tokenizer.chat_template = template
        prompts = tokenizer.apply_chat_template(
            orig_prompts, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if isinstance(prompts, str):
            prompts = eval(prompts)
        template_original_prompts = dict(zip(prompts, orig_prompts))
    else:
        prompts = orig_prompts
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions, add_special_tokens=False).input_ids
    skip_special_tokens = True
    if hasattr(tokenizer, "eod_id") and any(
        [token_id > tokenizer.eod_id for token_id in completion_token_ids[0]]
    ):
        skip_special_tokens = False
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        completion = tokenizer.decode(
            completion_token_ids[i][:output_len],
            skip_special_tokens=skip_special_tokens,
        )
        tokenized_dataset.append(
            (prompts[i], completion, prompt_token_ids[i], output_len)
        )
    return tokenized_dataset, template_original_prompts


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
    chat_template: Optional[str] = None,
    add_generation_prompt: Optional[bool] = False,
) -> List[Tuple[str, str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    tokenized_dataset, template_original_prompts = read_dataset(
        dataset_path,
        num_requests,
        tokenizer,
        fixed_output_len,
        chat_template,
        add_generation_prompt,
    )

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, str, int, int]] = []
    for prompt, completion, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        filtered_dataset.append((prompt, completion, prompt_len, output_len))

    # Sample the requests.
    num_requests = (
        num_requests if num_requests < len(filtered_dataset) else len(filtered_dataset)
    )
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests, template_original_prompts


def demo_requests(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    output_len: int,
    chat_template: Optional[str] = None,
    add_generation_prompt: Optional[bool] = False,
) -> List[Tuple[str, str, int, int]]:
    import demo_prompt

    template_original_prompts = None
    orig_prompts = getattr(demo_prompt, TASK_MAP[task].replace("-", "_"))
    if chat_template is not None:
        if chat_template != "default":
            with open(chat_template, "r") as f:
                template = f.read()
            tokenizer.chat_template = template
        prompts = tokenizer.apply_chat_template(
            orig_prompts, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        if isinstance(prompts, str):
            prompts = eval(prompts)
        template_original_prompts = dict(zip(prompts, orig_prompts))
    else:
        prompts = orig_prompts
    prompt_token_ids = tokenizer(prompts).input_ids
    demo_requests = [
        (prompt, None, len(prompt_token_id), output_len)
        for prompt, prompt_token_id in zip(prompts, prompt_token_ids)
    ]
    return demo_requests, template_original_prompts


def fake_requests(
    input_len: int,
    output_len: int,
    num_prompts: int,
    random_prompt: bool,
    dataset_for_perf: str,
    chat_template: Optional[str],
    add_generation_prompt: Optional[bool],
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:

    if random_prompt:
        fake_requests = []
        vocab_size = tokenizer.vocab_size
        for _ in range(num_prompts):
            # Synthesize a prompt with the given input length.
            candidate_ids = [
                random.randint(0, vocab_size - 1) for _ in range(args.input_len)
            ]
            # As tokenizer may add additional tokens like BOS, we need to try
            # different lengths to get the desired input length.
            while True:  # Max attempts to correct
                prompt = tokenizer.decode(candidate_ids)
                tokenized_len = len(tokenizer.encode(prompt))

                if tokenized_len == input_len:
                    break

                # Adjust length based on difference
                diff = input_len - tokenized_len
                if diff > 0:
                    candidate_ids.extend(
                        [random.randint(100, vocab_size - 100) for _ in range(diff)]
                    )
                else:
                    candidate_ids = candidate_ids[:diff]
            fake_requests.append((prompt, None, input_len, output_len))
    elif dataset_for_perf:
        tokenized_dataset, _ = read_dataset(
            dataset_for_perf,
            num_prompts,
            tokenizer,
            output_len,
            chat_template,
            add_generation_prompt,
        )

        # Filter out too long sequences.
        vocab_size = tokenizer.vocab_size
        filtered_dataset: List[Tuple[str, str, int, int]] = []
        for prompt, completion, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < input_len:
                # Prune too short sequences.
                continue
            else:
                candidate_ids = prompt_token_ids[:input_len]

                while True:  # Max attempts to correct
                    prompt = tokenizer.decode(candidate_ids)
                    tokenized_len = len(tokenizer.encode(prompt))

                    if tokenized_len == input_len:
                        break

                    # Adjust length based on difference
                    diff = input_len - tokenized_len
                    if diff > 0:
                        candidate_ids.extend(
                            [random.randint(100, vocab_size - 100) for _ in range(diff)]
                        )
                    else:
                        candidate_ids = candidate_ids[:diff]
                filtered_dataset.append((prompt, None, input_len, output_len))

        if len(filtered_dataset) < num_prompts:
            raise ValueError(
                "the dataset does not contain enough \
                             prompts that exceed the specified length."
            )

        fake_requests = random.sample(filtered_dataset, num_prompts)
        # for fake_prompt,_,_,_ in fake_requests:
        #     print(f"fake_prompt:{fake_prompt}")
    else:
        prompt = "hi" * input_len
        special_tokens_len = len(tokenizer(prompt).input_ids) - input_len
        if special_tokens_len > 0:
            prompt = "hi" * (input_len - special_tokens_len)
        fake_requests = [
            (prompt, None, input_len, output_len) for _ in range(num_prompts)
        ]
    return fake_requests


async def run_vllm_async(
    requests: List[Tuple[str, int, int]],
    n: int,
    engine_args: AsyncEngineArgs,
    disable_frontend_multiprocessing: bool = False,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    num_iters: int = 1,
    ignore_eos=True,
    profile=False,
) -> float:

    async with build_async_engine_client_from_engine_args(
        engine_args, disable_frontend_multiprocessing
    ) as llm:

        # Add the requests to the engine.
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        for prompt, _, _, output_len in requests:
            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=n,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    ignore_eos=ignore_eos,
                    max_tokens=output_len,
                )
            )

        generators = []
        if profile:
            llm.start_profile()
        start = time.perf_counter()
        for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
            generator = llm.generate(prompt, sp, request_id=f"test{i}")
            generators.append(generator)
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        end = time.perf_counter()
        if profile:
            llm.stop_profile()
        return end - start, None, []


def run_vllm(
    requests: List[Tuple[str, int, int]],
    n: int,
    engine_args: AsyncEngineArgs,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    num_iters: int = 1,
    ignore_eos=True,
    profile=False,
) -> Tuple[float, Optional[float], List["RequestOutput"]]:

    # from vllm_utils.dpllm import RayLLMWrapper
    # engine = RayLLMWrapper(engine_args)
    engine = LLMEngine.from_engine_args(engine_args)
    request_counter = Counter()

    dummy_sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
        max_tokens=1,
    )

    engine.add_request(
        str(next(request_counter)),
        prompt=requests[0][0],
        params=dummy_sampling_params,
    )
    while engine.has_unfinished_requests():
        _: list[RequestOutput] = engine.step()

    if profile:
        engine.start_profile()

    avg_decode_latency = 0

    start = time.perf_counter()
    for i in range(num_iters):
        # Add the requests to the engine.
        for _, (prompt, _, _, output_len) in enumerate(requests):
            sampling_params = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                max_tokens=output_len,
            )
            engine.add_request(
                str(next(request_counter)), prompt=prompt, params=sampling_params
            )

        use_tqdm = True

        if use_tqdm:
            num_requests = engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(
                    f"est. speed input: {0:.2f} toks/s, " f"output: {0:.2f} toks/s"
                ),
            )

        outputs: List[Union[RequestOutput, PoolingRequestOutput]] = []
        timings = {}

        total_in_toks = 0
        total_out_toks = 0

        while engine.has_unfinished_requests():
            step_outputs = engine.step()

            if isinstance(engine, V1Engine):
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
                # add metrics
                continue
            else:
                for output in step_outputs:
                    req_id = int(output.request_id)

                    if req_id not in timings:
                        timings.update(
                            {req_id: {"prefill": 0.0, "decode": 0.0, "last": 0.0}}
                        )

                    if len(output.outputs[0].token_ids) == 1:
                        timings[req_id]["prefill"] = (
                            output.metrics.first_token_time
                            - output.metrics.first_scheduled_time
                        )
                    else:
                        timings[req_id]["decode"] += (
                            output.metrics.last_token_time - timings[req_id]["last"]
                        )

                    timings[req_id]["last"] = output.metrics.last_token_time

                    if output.finished:
                        outputs.append(output)
                        if use_tqdm:
                            if isinstance(output, RequestOutput):
                                # Calculate tokens only for RequestOutput
                                assert output.prompt_token_ids is not None
                                total_in_toks += len(output.prompt_token_ids)
                                in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                total_out_toks += sum(
                                    len(stp.token_ids) for stp in output.outputs
                                )
                                out_spd = total_out_toks / pbar.format_dict["elapsed"]
                                pbar.postfix = (
                                    f"est. speed input: {in_spd:.2f} toks/s, "
                                    f"output: {out_spd:.2f} toks/s"
                                )
                            pbar.update(1)

        if use_tqdm:
            pbar.close()
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        outputs = LLMEngine.validate_outputs(outputs, RequestOutput)

        if not engine_args.disable_log_stats:
            timings = dict(sorted(timings.items(), key=lambda item: item[1]["decode"]))
            if timings:
                min_req_index = list(timings)[0]
                min_sum_decode_latency = timings[min_req_index]["decode"]
                min_req_outputs = list(filter(lambda x: int(x.request_id) == min_req_index, outputs))[0]
                real_decode_num = len(min_req_outputs.outputs[0].token_ids) - 1

                if real_decode_num > 0:
                    mean_decode_latency = min_sum_decode_latency / real_decode_num
                    avg_decode_latency += mean_decode_latency
        else:
            # disable_log_stats
            mean_decode_latency = None

    end = time.perf_counter()
    if profile:
        engine.stop_profile()

    return (end - start) / num_iters, avg_decode_latency / num_iters, outputs


def run_hf(
    requests: List[Tuple[str, str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
    trust_remote_code: bool,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, trust_remote_code=trust_remote_code
    )
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.perf_counter()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, _, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (
                max(max_prompt_len, next_prompt_len)
                + max(max_output_len, next_output_len)
            ) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.perf_counter()
    return end - start


def run_mii(
    requests: List[Tuple[str, str, int, int]],
    model: str,
    tensor_parallel_size: int,
    output_len: int,
) -> float:
    from mii import pipeline

    llm = pipeline(model, tensor_parallel=tensor_parallel_size)
    prompts = [prompt for prompt, _, _, _ in requests]

    start = time.perf_counter()
    llm(prompts, max_new_tokens=output_len)
    end = time.perf_counter()
    return end - start


def vllm_perf(
    requests: List[Tuple[str, str, int, int]],
    outputs: List["RequestOutput"],
    elapsed_time: float,
    # decode_elapsed_time: Optional[float],
    mean_avg_decode_latency: Optional[float],
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

    input_len = [_input_len for _, _, _input_len, _ in requests]
    output_len = [_output_len for _, _, _, _output_len in requests]
    max_output_len = max(output_len)
    total_num_tokens = sum(output_len)
    if mean_avg_decode_latency == 0:
        print("decode elapsed time is not recorded as set `--disable_log_stats`")
        decode_latency_per_token = None
        decode_throughput = None
    else:
        decode_throughput = f"{1 / mean_avg_decode_latency :.2f} tokens/s"
        decode_latency_per_token = f"{mean_avg_decode_latency * 1000 :.2f} ms"

    perf_info = dict(
        total_input_tokens=sum(input_len),
        total_output_tokens=total_num_tokens,
        latency_num_prompts=f"{elapsed_time * 1000:.2f} ms",
        latency_per_token=f"{elapsed_time * 1000 / max_output_len:.2f} ms",
        request_per_second=f"{len(requests) / elapsed_time:.2f} requests/s",
        token_per_second=f"{total_num_tokens / elapsed_time:.2f} tokens/s",
        prefill_latency_per_token=f"{prefill_latency_per_token * 1000:.2f} ms",
        decode_latency_per_token=decode_latency_per_token,
        decode_throughput=decode_throughput,
    )

    return perf_info


def vllm_acc(
    requests: List[Tuple[str, str, int, int]],
    outputs: List["RequestOutput"],
    tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, str]:
    import evaluate

    rouge_path = os.path.join(ROOT_DIR, "rouge.py")
    if os.path.exists(rouge_path):
        metric = evaluate.load(rouge_path)
    else:
        metric = evaluate.load("rouge")

    predictions = [output.outputs[0].text for output in outputs]
    references = [
        completion for _, completion, _, _ in requests if completion is not None
    ]
    predictions, references = zip(
        *[(i, j) for i, j in zip(predictions, references) if i != "" or j != ""]
    )
    assert len(predictions) == len(references), "acc is only valid for dataset"

    rouges = metric.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False,
        tokenizer=tokenizer.tokenize,
    )
    return {k: round(np.mean(v) * 100, 4) for k, v in rouges.items()}


def save_to_csv(csv_file, data):
    import csv

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        writer.writerows(data.items())

    return csv_file


def save_results(outputs, outfile, template_original_prompts):
    inference_results = []
    for output in outputs:
        curr_conversion = {}

        content = []
        prompt = {}
        save_prompt = output.prompt
        if template_original_prompts:
            save_prompt = template_original_prompts[save_prompt]
        prompt["value"] = save_prompt
        content.append(prompt)

        generate_text = {}
        generate_text["value"] = output.outputs[0].text
        content.append(generate_text)

        generate_tokens = {}
        token_ids = ",".join([str(elem) for elem in output.outputs[0].token_ids])
        generate_tokens["value"] = token_ids
        content.append(generate_tokens)

        curr_conversion["conversations"] = content
        inference_results.append(curr_conversion)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(json.dumps(inference_results, indent=4, ensure_ascii=False))


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    msgs = vars(args)

    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer = get_tokenizer(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=args.trust_remote_code
        )

    num_iters = 1
    if args.dataset:
        requests, template_original_prompts = sample_requests(
            args.dataset,
            args.num_prompts,
            tokenizer,
            args.output_len,
            args.template,
            args.add_generation_prompt,
        )
    elif args.demo:
        requests, template_original_prompts = demo_requests(
            args.demo,
            tokenizer,
            args.output_len,
            args.template,
            args.add_generation_prompt,
        )
    else:
        # Synthesize a prompt with the given input length.
        requests = fake_requests(
            args.input_len,
            args.output_len,
            args.num_prompts,
            args.random_prompt,
            args.dataset_for_perf,
            args.template,
            args.add_generation_prompt,
            tokenizer,
        )
        if args.output_len == 1:
            num_iters = 10

    if args.backend == "vllm":
        run_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "num_iters": num_iters,
            "ignore_eos": not args.acc and args.demo is None,
            "profile": args.profile,
        }

        if args.async_engine:
            args.disable_log_requests = True
            elapsed_time, mean_avg_decode_latency, outputs = uvloop.run(
                run_vllm_async(
                    requests,
                    args.n,
                    AsyncEngineArgs.from_cli_args(args),
                    args.disable_frontend_multiprocessing,
                    **run_kwargs,
                )
            )
        else:
            elapsed_time, mean_avg_decode_latency, outputs = run_vllm(
                requests, args.n, EngineArgs.from_cli_args(args), **run_kwargs
            )

        if args.perf:
            perf_info = vllm_perf(
                requests, outputs, elapsed_time, mean_avg_decode_latency
            )
            msgs.update(perf_info)

            print("\n***Perf Info***")
            print(json.dumps(perf_info, indent=4))

        if args.acc:
            assert args.dataset is not None, "only evaluate on dataset"
            acc_info = vllm_acc(requests, outputs, tokenizer)
            msgs.update(acc_info)

            print("\n***Acc Info***")
            print(json.dumps(acc_info, indent=4))

        if args.demo:
            for request, output in zip(requests, outputs):
                print(
                    f"Prompt: {request[0]!r}, "
                    f"Generated text: {output.outputs[0].text!r}"
                )
        else:
            if args.save_output:
                save_results(outputs, args.save_output, template_original_prompts)
            csvfile = save_to_csv(f"{time.strftime('%Y%m%d%H%M%S')}.csv", msgs)
            print(f"save to {csvfile}")

    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(
            requests,
            args.model,
            tokenizer,
            args.n,
            args.use_beam_search,
            args.hf_max_batch_size,
            args.trust_remote_code,
        )
    elif args.backend == "mii":
        elapsed_time = run_mii(
            requests, args.model, args.tensor_parallel_size, args.output_len
        )
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len for _, _, prompt_len, output_len in requests
    )
    print(
        f"Request Latency: {elapsed_time / len(requests):.2f} s, "
        f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
        f"{total_num_tokens / elapsed_time:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark test.")
    parser.add_argument(
        "--backend", type=str, choices=["vllm", "hf", "mii"], default="vllm"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=None,
        help="Input prompt length for each request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the "
        "output length from the dataset.",
    )
    parser.add_argument(
        "--n", type=int, default=1, help="Number of generated sequences per prompt."
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--hf-max-batch-size",
        type=int,
        default=None,
        help="Maximum batch size for HF backend.",
    )
    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="file to save dataset inference results",
    )
    parser.add_argument(
        "--async-engine",
        action="store_true",
        default=False,
        help="Use vLLM async engine rather than LLM class.",
    )
    parser.add_argument(
        "--disable-frontend-multiprocessing",
        action="store_true",
        default=False,
        help="Disable decoupled async engine frontend.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature for sampling."
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Float that controls the cumulative probability of the top tokens to consider. \
            Must be in (0, 1]. Set to 1 to consider all tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Integer that controls the number of top tokens to consider. \
            Set to -1 to consider all tokens.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Float that penalizes new tokens based on whether \
            they appear in the prompt and the generated text so far. Values > 1 \
            encourage the model to use new tokens, while values < 1 encourage \
            the model to repeat tokens.",
    )
    parser.add_argument(
        "--demo", type=str, default=None, choices=TASK_MAP.keys(), help=f"{TASK_MAP}"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="either 'default' or path to template for tokenizer, if 'default', use default chat template of tokenizer",
    )
    parser.add_argument(
        "--add-generation-prompt", type=bool, default=False, help="add-generation-promp"
    )
    parser.add_argument("--perf", action="store_true", help="readout perf")
    parser.add_argument(
        "--random-prompt", action="store_true", help="use real random prompts"
    )
    parser.add_argument(
        "--dataset-for-perf",
        type=str,
        default=None,
        help="Path to the dataset for perf.",
    )
    parser.add_argument("--acc", action="store_true", help="evaluate on dataset")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. Must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler."
        "gcu only support host events now.",
    )
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    if args.perf:
        assert not (
            args.random_prompt and (args.dataset_for_perf is not None)
        ), "Cannot enable both random-prompt mode and dataset-for-perf mode simultaneously"
    if args.async_engine:
        print("Warning: async engine is not supported completely")

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.device == "gcu":
        args.trust_remote_code = True
    if args.dataset:
        assert args.input_len is None
    elif args.demo:
        assert args.input_len is None
        assert args.output_len is not None
        assert not args.acc
    else:
        assert args.input_len is not None
        assert args.output_len is not None
        assert not args.acc

    if args.template and args.template != "default":
        if not os.path.exists(args.template):
            args.template = os.path.join(ROOT_DIR, args.template)
            assert os.path.exists(args.template), f"{args.template} not exists"

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")

        additional_config = args.additional_config if args.additional_config else {}
        if args.enable_chunked_prefill:
            additional_config.update({"enable_chunked_prefill": True})
        if args.enable_prefix_caching:
            additional_config.update({"enable_prefix_caching": True})
        if args.max_num_batched_tokens:
            additional_config.update({"max_num_batched_tokens": args.max_num_batched_tokens})
        args.additional_config = additional_config
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.max_num_seqs is not None:
            raise ValueError("max_num_seqs is only for vLLM backend.")
    elif args.backend == "mii":
        if args.dtype != "auto":
            raise ValueError("dtype must be auto for MII backend.")
        if args.n != 1:
            raise ValueError("n must be 1 for MII backend.")
        if args.use_beam_search:
            raise ValueError("Beam search is not supported for MII backend.")
        if args.quantization is not None:
            raise ValueError("Quantization is only for vLLM backend.")
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
        if args.tokenizer != args.model:
            raise ValueError(
                "Tokenizer must be the same as the model for MII " "backend."
            )
        if args.max_num_seqs is not None:
            raise ValueError("max_num_seqs is only for vLLM backend.")

    # TODO: add lora and multi modal
    main(args)
