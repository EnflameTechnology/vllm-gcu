import argparse
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
    parser.add_argument('--max-num-batched-tokens', type=int, default=256)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="disable eager execution")
    parser.add_argument("--enable-chunked-prefill",
                        action="store_true",
                        help="enable chunked prefill")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # Create an LLM.
    llm = LLM(model=args.model,enable_chunked_prefill=args.enable_chunked_prefill,
              device=args.device,enforce_eager=args.enforce_eager,
              tensor_parallel_size = args.tensor_parallel_size,
              gpu_memory_utilization= args.gpu_memory_utilization,
              max_num_batched_tokens=args.max_num_batched_tokens)

    generating_prompts = [prefix + prompt for prompt in prompts]

    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(generating_prompts, sampling_params)

    # save the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
