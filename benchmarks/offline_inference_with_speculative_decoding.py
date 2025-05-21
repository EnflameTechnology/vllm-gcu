import os
import sys
import argparse
import time
import json
import multiprocessing
from typing import List, Optional, Tuple
from transformers import (AutoTokenizer, PreTrainedTokenizerBase)

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
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--speculative-model', type=str, default='JackFram/llama-68m')
    parser.add_argument('--device', type=str, default='gcu', choices=['gcu', 'cuda'])
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--num-speculative-tokens', type=int, default=5)
    parser.add_argument('--ngram-prompt-lookup-max', type=int, default=3)
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="enable eager execution")
    parser.add_argument("--add-generation-prompt",
                        type=bool,
                        default=False,
                        help="add-generation-promp")
    parser.add_argument("--template",
                        type=str,
                        default=None,
                        help="either 'default' or path to template for tokenizer, \
                            if 'default', use default chat template of tokenizer")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="trust remote code")
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument("--demo", type=str, default=None,
                        choices=TASK_MAP.keys(),
                        help=f"{TASK_MAP}")
    parser.add_argument("--save-output",
                        type=str,
                        default="inference_results_with_speculative.json",
                        help="file to save dataset inference results")
    return parser.parse_args()

def demo_prompts(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
    chat_template: Optional[str] = None,
) -> List[Tuple[str, str, int, int]]:
    import demo_prompt
    orig_prompts = getattr(demo_prompt, TASK_MAP[task].replace("-", "_"))
    if chat_template is not None:
        if chat_template != "default":
            with open(chat_template, "r") as f:
                template = f.read()
            tokenizer.chat_template = template
        prompts = tokenizer.apply_chat_template(
            orig_prompts, tokenize=False, add_generation_prompt=add_generation_prompt)
        if isinstance(prompts, str):
            prompts = eval(prompts)
    else:
        prompts = orig_prompts

    return prompts

def get_llm(llm_type:str,
            args:argparse.Namespace) :
    from vllm import LLM
    if llm_type == "default":
        llm = LLM(model=args.model,
              device=args.device,
              enforce_eager=args.enforce_eager,
              tensor_parallel_size = args.tensor_parallel_size,
              gpu_memory_utilization= args.gpu_memory_utilization)
    elif llm_type == "speculative":
        llm = LLM(model=args.model,
                speculative_model = args.speculative_model,
                device=args.device,enforce_eager=args.enforce_eager,
                tensor_parallel_size = args.tensor_parallel_size,
                gpu_memory_utilization= args.gpu_memory_utilization,
                num_speculative_tokens=args.num_speculative_tokens,
                use_v2_block_manager=True)
    elif llm_type == "ngram":
        llm = LLM(model=args.model,
                speculative_model = "[ngram]",
                device=args.device,enforce_eager=args.enforce_eager,
                tensor_parallel_size = args.tensor_parallel_size,
                gpu_memory_utilization= args.gpu_memory_utilization,
                num_speculative_tokens=args.num_speculative_tokens,
                ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
                use_v2_block_manager=True)
    else:
        print(f"unsupported llm type:{llm_type}")
        llm = None

    return llm

def inference(llm_type, args, queue):
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(
            args.model, trust_remote_code=args.trust_remote_code)
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code)

    prompts = demo_prompts(args.demo, tokenizer,
                           args.add_generation_prompt,
                           args.template)

    # prompts = "The future of AI is"
    # Create an LLM
    llm = get_llm(llm_type, args)

    # Generate texts from the prompts
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapse_time = (time.time() - start_time) * 1000

    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(f"prompt:{prompt},generate text:{generated_text}")

    inference_results = {}
    inference_results["elapse_time"] = elapse_time
    inference_results["generated_texts"] = generated_texts

    queue.put({llm_type:inference_results})

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    llm_types = ["default", "speculative", "ngram"]

    all_results = {}
    queue = multiprocessing.Queue(maxsize=len(llm_types))
    for llm_type in llm_types:
        process = multiprocessing.Process(target=inference, args=(llm_type, args, queue))
        process.start()
        process.join()
        all_results.update(queue.get())

    with open(args.save_output, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    for key,value in all_results.items():
        if key == "default":
            base_elapse_time = value["elapse_time"]
            base_texts = value["generated_texts"]
        else:
            elapse_time = value["elapse_time"]
            texts = value["generated_texts"]

            compare_result = "mismatch"
            if texts == base_texts:
                compare_result = "match"

            speedup_ratio = base_elapse_time/elapse_time

            print(f"inference type:{key}, inference result:{compare_result}, speedup ratio:{speedup_ratio:.2f}")
