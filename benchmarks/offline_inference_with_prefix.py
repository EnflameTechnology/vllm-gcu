import argparse
import time
import json
from vllm import LLM, SamplingParams

prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--device', type=str, default='gcu', choices=['gcu', 'cuda'])
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="disable eager execution")
    parser.add_argument("--save-output",
                        type=str,
                        default="inference_results_with_prefix.json",
                        help="file to save dataset inference results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Create an LLM.
    llm = LLM(model=args.model,enable_prefix_caching=True, \
              device=args.device,enforce_eager=args.enforce_eager, \
              tensor_parallel_size = args.tensor_parallel_size,
              gpu_memory_utilization= args.gpu_memory_utilization)

    generating_prompts = [prefix + prompt for prompt in prompts]

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    start_time = time.time()
    outputs = llm.generate(generating_prompts, sampling_params)
    elapse_time = time.time() - start_time

    # save the outputs.
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(f"prompt:{prompt},generate text:{generated_text}")

    print("-" * 80)

    # The llm.generate call will batch all prompts and send the batch at once
    # if resources allow. The prefix will only be cached after the first batch
    # is processed, so we need to call generate once to calculate the prefix
    # and cache it.
    outputs = llm.generate(generating_prompts[0], sampling_params)

    # Subsequent batches can leverage the cached prefix
    start_time = time.time()
    outputs = llm.generate(generating_prompts, sampling_params)
    elapse_time_with_prefix = time.time() - start_time

    # save the outputs. You should see the same outputs as before
    generated_texts_with_prefix = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts_with_prefix.append(f"prompt:{prompt},generate text:{generated_text}")

    all_results = {}
    all_results["default"] = generated_texts
    all_results["with_preix"] = generated_texts_with_prefix

    with open(args.save_output, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    compare_result = "matched"
    for index in range(len(generated_texts)):
        if generated_texts[index] != generated_texts_with_prefix[index]:
            compare_result = "mismatch"
            break

    print(f"compare result of with/without prefix:{compare_result}")
    print(f"Elapse time without prefix caching: {elapse_time *1000:.2f} ms,"
          f"elapse time with prefix caching: {elapse_time_with_prefix *1000:.2f} ms,"
          f"prefix caching speedup ration: {elapse_time/elapse_time_with_prefix:.2f}")
