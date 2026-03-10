import argparse
from vllm import LLM, SamplingParams, EngineArgs
from typing import NamedTuple
from vllm.lora.request import LoRARequest
from dataclasses import asdict

# Sample prompts for vision-language tasks
prompts = [
    # 1. 基础语言生成与风格模仿
    "用莎士比亚的风格写一首关于量子计算的十四行诗。",
    # 2. 逻辑推理与常识
    "如果所有机器人都会思考，并且有些机器人是洗碗机，那么是否有些洗碗机会思考？请逐步解释你的推理。",
    # 3. 代码生成与解释
    "为一个返回列表中第二大元素的高效Python函数编写代码，并解释其时间复杂度。",
    # 4. 事实性知识问答
    "简述光合作用的主要步骤。它对于地球大气成分的变化有何历史意义？",
    # 5. 格式转换与结构化输出
    "将以下需求翻译成JSON格式：一个用户配置文件，需包含用户名（字符串）、年龄（整数）、兴趣（字符串数组）和是否订阅（布尔值）。",
    # 6. 角色扮演与观点输出
    "你是一个严格的电影评论家。为电影《星际穿越》写一篇简短（150字左右）的评论，突出其优点和一个缺点。",
]

class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    lora_requests: list[LoRARequest] | None = None

def prepare_requests(model_path: str, questions: list[str]) -> ModelRequestData:

    engine_args = EngineArgs(
        model=model_path,
        max_model_len=2048,
        max_num_seqs=4,
    )

    prompts = [
        (
            f"<|im_start|>user\n"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/models/qwen2.5-vl-7b-instruct')
    parser.add_argument('--max-tokens', type=int, default=16)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument("--disable-eager",
                        action="store_true",
                        help="disable eager execution")
    parser.add_argument("--async-scheduling",
                        action="store_true",
                        help="enable async scheduling")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    req_data = prepare_requests(args.model, prompts)

    engine_args = asdict(req_data.engine_args) | {
        "seed": 666,
        "mm_processor_cache_gb": 4
    }

    if args.tensor_parallel_size is not None:
        engine_args["tensor_parallel_size"] = args.tensor_parallel_size
    
    engine_args["gpu_memory_utilization"] = args.gpu_memory_utilization
    engine_args["enforce_eager"] = not args.disable_eager
    engine_args["async_scheduling"] = args.async_scheduling

    llm = LLM(
        **engine_args
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    inputs = []
    for i, prompt in enumerate(req_data.prompts):
        inputs.append(
            {
                "prompt": prompts[i % len(prompts)],
            }
        )

    # Generate texts from the multimodal inputs
    outputs = llm.generate(inputs, sampling_params)

    # Save and display the outputs
    for i, output in enumerate(outputs):
        prompt_text = prompts[i]
        generated_text = output.outputs[0].text if output.outputs else "No output"
        
        print(f"\n{'='*50}")
        print(f"Prompt {i+1}: {prompt_text}")
        print(f"Generated text: {generated_text}")

        print('='*50)
