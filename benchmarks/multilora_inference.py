"""
This example shows how to use the multi-LoRA functionality for offline inference.

"""
import os
import argparse
import json
from typing import Optional, List, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

def get_lora_path(lora_models):
    lora_info = {}
    for lora_model in lora_models:
        if not os.path.exists(lora_model["model_path"]):
            lora_path = snapshot_download(repo_id=lora_model["model_path"])
            lora_model["model_path"] = lora_path
        lora_info[lora_model["id"]]=lora_model["model_path"]
    return lora_info

def create_requests(prompts,lora_models):
    lora_info = get_lora_path(lora_models)
    requests = []
    for prompt in prompts:
        request = []
        text = prompt["text"]
        request.append(text)
        lora_request = None
        if "lora_id" in prompt.keys():
            lora_id = int(prompt["lora_id"])
            lora_path = lora_info[lora_id]
            lora_request = LoRARequest("",lora_int_id=lora_id,lora_local_path=lora_path)
        request.append(lora_request)
        requests.append(request)
    return requests

def process_requests(engine: LLMEngine,
                     test_requests: List[Tuple[str,Optional[LoRARequest]]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    sampling_params = SamplingParams(temperature=0.0,
                        top_p=1.0,
                        max_tokens=128,
                        stop_token_ids=[32003])

    while test_requests or engine.has_unfinished_requests():
        if test_requests:
            prompt, lora_request = test_requests.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for request_output in request_outputs:
            if request_output.finished:
                print(f"Prompt: {request_output.prompt!r}, Generated text: {request_output.outputs[0].text!r}")


def initialize_engine(args) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model=args.base_model,
                             device=args.device,
                             trust_remote_code=True,
                             enable_lora=True,
                             max_loras=1,
                             max_lora_rank=8,
                             max_cpu_loras=2,
                             max_num_seqs=256,
                             enforce_eager=args.enforce_eager)
    return LLMEngine.from_engine_args(engine_args)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--device', type=str, default='gcu', choices=['gcu', 'cuda', 'cpu'],)
    parser.add_argument('--lora-config', type=str, default='yard1/llama-2-7b-sql-lora-test')
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enforce eager execution")
    return parser.parse_args()

def main(args):
    """Main function that sets up and runs the prompt processing."""

    with open(args.lora_config) as f:
        inputs = json.load(f)
        lora_models = inputs['lora_models']
        prompts = inputs['prompts']

    test_requests = create_requests(prompts,lora_models)

    engine = initialize_engine(args)
    process_requests(engine, test_requests)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
