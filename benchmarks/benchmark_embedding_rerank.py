import asyncio
import random
import time
import httpx
import numpy as np
import argparse
import json
from typing import List, Tuple, Any, Dict
from vllm_utils.benchmark_utils import write_to_json
from vllm.transformers_utils.tokenizer import get_tokenizer
from asyncio import Semaphore
from typing import AsyncGenerator, Optional
from tqdm.asyncio import tqdm  # Import progress bar for async tasks

# ------------------------------ Request Generator ------------------------------
async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    """Asynchronously generate requests with specified rate and burstiness"""
    input_requests = iter(input_requests)
    assert burstiness > 0, "Burstiness must be a positive value"

    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue

        theta = 1.0 / (request_rate * burstiness)
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)

# ------------------------------ Embedding Request ------------------------------
async def send_embedding_request(
    session: httpx.AsyncClient,
    prompt: str,
    api_url: str,
    headers: dict,
    model: str,
) -> Tuple[float, float | None]:  # Returns (start_time, latency)
    """Send a single embedding request and measure start time and latency"""
    start_time = time.perf_counter()
    try:
        response = await session.post(api_url, headers=headers, json={"model": model, "input": [prompt]})
        response.raise_for_status()
        end_time = time.perf_counter()
        return (start_time, end_time - start_time)
    except httpx.HTTPError:
        return (start_time, None)

# ------------------------------ Rerank Request ------------------------------
async def send_rerank_request(
    session: httpx.AsyncClient,
    query: str,
    documents: list,
    api_url: str,
    headers: dict,
    model: str,
) -> Tuple[float, float | None]:  # Returns (start_time, latency)
    """Send a single rerank request and measure start time and latency"""
    start_time = time.perf_counter()
    try:
        response = await session.post(api_url, headers=headers, json={"model": model, "query": query, "documents": documents})
        response.raise_for_status()
        end_time = time.perf_counter()
        return (start_time, end_time - start_time)
    except httpx.HTTPError as e:
        print(f"Request failed: {str(e)}")
        return (start_time, None)

# ------------------------------ Performance Calculation ------------------------------
def calculate_p99_and_avg(response_times: list) -> tuple[float | None, float | None]:
    """Calculate P99 and average response time from latency list"""
    if not response_times:
        return None, None
    valid_response_times = [rt for rt in response_times if rt is not None]
    valid_response_times.sort()
    p99 = valid_response_times[int(0.99 * len(valid_response_times))] if len(valid_response_times) > 1 else valid_response_times[0] if valid_response_times else None
    avg = sum(valid_response_times) / len(valid_response_times) if valid_response_times else None
    return round(p99, 5) if p99 is not None else None, round(avg, 5) if avg is not None else None

# ------------------------------ Random Prompt Generation ------------------------------
def generate_random_prompt(
    length: int,
    tokenizer: Any,
    num_prompts: int = 1,
    strict_in_out_len: bool = False,
) -> List[str]:
    """Generate random prompts with specified length"""
    prompts = []
    for _ in range(num_prompts):
        if strict_in_out_len:
            prompt = "hi" * length
        else:
            token_ids = np.random.randint(0, tokenizer.vocab_size, size=length).tolist()
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        prompts.append(prompt)
    return prompts

def generate_random_token(
    length: int,
    tokenizer: Any,
    num_prompts: int = 1,
) -> List[int]:
    prompts = []
    for _ in range(num_prompts):
        token_ids = np.random.randint(0, tokenizer.vocab_size, size=length).tolist()
        prompts.append(token_ids)
    return prompts

# ------------------------------ Embedding Test Main Function ------------------------------
async def test_embedding_requests(
    api_url: str,
    headers: dict,
    model: str,
    input_len: int,
    total_requests: int,
    request_rate: float,
    burstiness: float = 1.0,
    max_concurrency: Optional[int] = None,  # Default to None for unbounded concurrency
    save_to_file: bool = True,
    file_path: str = "performance_report.json",
    seed: int = 42,
    random_doc_len_enabled: bool = False,
    tokenizer: str = None,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    connect_timeout: float = 50.0,
    read_timeout: float = 1000.0,
    write_timeout: float = 50.0,
    pool_timeout: float = None,
    disable_tqdm: bool = False,  # Disable progress bar
) -> Dict[str, Any]:
    """Asynchronously test embedding API performance with concurrency control and progress bar"""
    random.seed(seed)
    np.random.seed(seed)

    # Initialize tokenizer
    tokenizer_id = tokenizer
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code
    )

    # Generate request list
    if random_doc_len_enabled:
        input_requests = [
            (generate_random_token(np.random.randint(1, input_len), tokenizer)[0],
             np.random.randint(1, input_len), 0)
            for _ in range(total_requests)
        ]
    else:
        prompts = generate_random_token(input_len, tokenizer, num_prompts=total_requests)
        input_requests = [(prompt, input_len, 0) for prompt in prompts]

    # Configure timeouts
    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )

    # Concurrency control
    semaphore = Semaphore(max_concurrency) if max_concurrency is not None else None

    async with httpx.AsyncClient(timeout=timeout) as session:
        _, _ = await send_embedding_request(session, input_requests[0][0], api_url, headers, model)
        start_all_time = time.perf_counter()
        tasks = []
        request_timestamps = []
        progress_bar = tqdm(total=total_requests, disable=disable_tqdm, desc="Embedding Requests")  # Initialize progress bar

        async for prompt, _, _ in get_request(input_requests, request_rate, burstiness):
            async def handle_request():
                nonlocal progress_bar  # Access progress bar in async context
                if semaphore:
                    async with semaphore:
                        start, latency = await send_embedding_request(session, prompt, api_url, headers, model)
                else:
                    start, latency = await send_embedding_request(session, prompt, api_url, headers, model)
                request_timestamps.append((start, latency))
                progress_bar.update(1)  # Update progress bar on request completion

            task = asyncio.create_task(handle_request())
            tasks.append(task)

        await asyncio.gather(*tasks)
        end_all_time = time.perf_counter()
        progress_bar.close()  # Close progress bar after all tasks complete

    # Process request timestamps
    valid_timestamps = [(start, lat) for start, lat in request_timestamps if lat is not None]
    if not valid_timestamps:
        first_start_time = None
        last_end_time = None
    else:
        valid_timestamps.sort(key=lambda x: x[0])
        first_start_time = valid_timestamps[0][0]
        last_end_time = valid_timestamps[-1][0] + valid_timestamps[-1][1]  # start_time + latency = end_time

    # Calculate avg_latency_per_sample
    avg_latency_per_sample = None
    if first_start_time is not None and last_end_time is not None and total_requests > 0:
        total_duration = last_end_time - first_start_time
        avg_latency_per_sample = total_duration / total_requests

    # Calculate other metrics
    response_times = [lat for start, lat in valid_timestamps]
    p99, avg_response_time = calculate_p99_and_avg(response_times)
    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency or 'Resource-Bounded Concurrency'}")

    # Build performance report
    report = {
        "test_type": "embedding",
        "api_url": api_url,
        "model": model,
        "input_len": input_len,
        "total_requests": total_requests,
        "request_rate": request_rate,
        "burstiness": burstiness,
        "random_doc_len_enabled": random_doc_len_enabled,
        "success_requests": len(response_times),
        "p99(s)": p99,
        "average_response_time(s)": avg_response_time,
        "total_time(s)": round(end_all_time - start_all_time, 5),
        "seed": seed,
        "tokenizer": tokenizer_id,
        "tokenizer_mode": tokenizer_mode,
        "connect_timeout(s)": connect_timeout,
        "read_timeout(s)": read_timeout,
        "write_timeout(s)": write_timeout,
        "pool_timeout(s)": pool_timeout,
        "max_concurrency": max_concurrency,
        "avg_latency_per_sample(s)": round(avg_latency_per_sample, 5) if avg_latency_per_sample is not None else None,
        "encode_token_speed" : f'{(input_len * len(response_times)) / (end_all_time - start_all_time)} token/s'
    }

    print(json.dumps(report, indent=4))
    if save_to_file:
        write_to_json(file_path.replace(".json", "_embedding.json"), report)
    return report

# ------------------------------ Rerank Test Main Function ------------------------------
async def test_rerank_requests(
    api_url: str,
    headers: dict,
    model: str,
    input_len: int,
    total_requests: int,
    request_rate: float,
    burstiness: float = 1.0,
    max_concurrency: Optional[int] = None,  # Default to None for unbounded concurrency
    query_len: int = 15,
    num_docs: int = 10,
    save_to_file: bool = True,
    file_path: str = "performance_report.json",
    seed: int = 42,
    random_doc_len_enabled: bool = False,
    tokenizer: str = None,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    connect_timeout: float = 50.0,
    read_timeout: float = 1000.0,
    write_timeout: float = 50.0,
    pool_timeout: float = None,
    disable_tqdm: bool = False,  # Disable progress bar
) -> Dict[str, Any]:
    """Asynchronously test rerank API performance with concurrency control and progress bar"""
    random.seed(seed)
    np.random.seed(seed)


    # Initialize tokenizer
    tokenizer_id = tokenizer
    tokenizer = get_tokenizer(
        tokenizer_id,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code
    )

    # Generate query and documents
    query = generate_random_prompt(query_len, tokenizer)[0]
    if random_doc_len_enabled:
        documents = [generate_random_prompt(np.random.randint(1, input_len), tokenizer)[0] for _ in range(num_docs)]
    else:
        documents = generate_random_prompt(input_len, tokenizer, num_prompts=num_docs)

    input_requests = [(query, len(documents), 0) for _ in range(total_requests)]

    # Configure timeouts
    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )

    # Concurrency control
    semaphore = Semaphore(max_concurrency) if max_concurrency is not None else None

    async with httpx.AsyncClient(timeout=timeout) as session:
        _, _ = await send_rerank_request(session, input_requests[0][0], documents[:input_requests[0][1]], api_url, headers, model)
        tasks = []
        request_timestamps = []
        progress_bar = tqdm(total=total_requests, disable=disable_tqdm, desc="Rerank Requests")  # Initialize progress bar
        start_all_time = time.perf_counter()  # Record global start time
        async for query_text, doc_count, _ in get_request(input_requests, request_rate, burstiness):
            async def handle_request():
                nonlocal progress_bar  # Access progress bar in async context
                if semaphore:
                    async with semaphore:
                        start, latency = await send_rerank_request(session, query_text, documents[:doc_count], api_url, headers, model)
                else:
                    start, latency = await send_rerank_request(session, query_text, documents[:doc_count], api_url, headers, model)
                request_timestamps.append((start, latency))
                progress_bar.update(1)  # Update progress bar on request completion

            task = asyncio.create_task(handle_request())
            tasks.append(task)

        await asyncio.gather(*tasks)
        end_all_time = time.perf_counter()
        progress_bar.close()  # Close progress bar after all tasks complete

    # Process request timestamps
    valid_timestamps = [(start, lat) for start, lat in request_timestamps if lat is not None]
    if not valid_timestamps:
        first_start_time = None
        last_end_time = None
    else:
        valid_timestamps.sort(key=lambda x: x[0])
        first_start_time = valid_timestamps[0][0]
        last_end_time = valid_timestamps[-1][0] + valid_timestamps[-1][1]  # start_time + latency = end_time

    # Calculate avg_latency_per_sample
    avg_latency_per_sample = None
    if first_start_time is not None and last_end_time is not None and total_requests > 0:
        total_duration = last_end_time - first_start_time
        avg_latency_per_sample = total_duration / total_requests

    # Calculate other metrics
    response_times = [lat for start, lat in valid_timestamps]
    p99, avg_response_time = calculate_p99_and_avg(response_times)
    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency or 'Resource-Bounded Concurrency'}")

    # Build performance report
    report = {
        "test_type": "rerank",
        "api_url": api_url,
        "model": model,
        "input_len": input_len,
        "total_requests": total_requests,
        "request_rate": request_rate,
        "burstiness": burstiness,
        "query_len": query_len,
        "num_docs": num_docs,
        "random_doc_len_enabled": random_doc_len_enabled,
        "success_requests": len(response_times),
        "p99(s)": p99,
        "average_response_time(s)": avg_response_time,
        "total_time(s)": round(end_all_time - start_all_time, 5),
        "seed": seed,
        "tokenizer": tokenizer_id,
        "tokenizer_mode": tokenizer_mode,
        "connect_timeout(s)": connect_timeout,
        "read_timeout(s)": read_timeout,
        "write_timeout(s)": write_timeout,
        "pool_timeout(s)": pool_timeout,
        "max_concurrency": max_concurrency,
        "avg_latency_per_sample(s)": round(avg_latency_per_sample, 5) if avg_latency_per_sample is not None else None,
    }

    print(json.dumps(report, indent=4))
    if save_to_file:
        write_to_json(file_path.replace(".json", "_rerank.json"), report)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Testing Tool for Embedding and Rerank APIs')
    parser.add_argument('--test-type', type=str, choices=['embedding', 'rerank'], required=True,
                        help='Type of test to run, either "embedding" or "rerank"')
    parser.add_argument('--api-url', type=str, required=True, help='API URL')
    parser.add_argument('--headers', type=str, default='{"Content-Type": "application/json"}',
                        help='API headers as a JSON string, default is {"Content-Type": "application/json"}')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--input-len', type=int, default=500,
                        help='Input length (fixed or max random length when --random_doc_len_enabled is used)')
    parser.add_argument('--total-requests', type=int, default=100, help='Total number of requests to send')
    parser.add_argument('--request-rate', type=float, default=float('inf'),
                        help='Request rate in requests per second, use "inf" for immediate sending')
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for request generation. "
             "Valid when request_rate is not infinite. "
             "1.0 = Poisson process (default), <1 = bursty, >1 = more uniform.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This controls how many requests "
             "are allowed to execute simultaneously. Use 0 or omit for unlimited concurrency.",
    )
    parser.add_argument('--query-len', type=int, default=15, help='Length of the query text in tokens/characters')
    parser.add_argument('--num-docs', type=int, default=10, help='Number of documents per rerank request')
    parser.add_argument('--save-to-file', action='store_true', default=True,
                        help='Save performance report to JSON file (default: True)')
    parser.add_argument('--file-path', type=str, default='performance_report.json',
                        help='Base file path for saving reports')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible prompt generation')
    parser.add_argument('--random-doc-len-enabled', action='store_true', default=False,
                        help='Enable random input/document length (1 to input_len)')

    # Tokenizer configuration
    parser.add_argument('--tokenizer', type=str, help='Tokenizer name or path (e.g., "bert-base-uncased")')
    parser.add_argument(
        '--tokenizer-mode',
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help='Tokenizer mode:\n'
             'auto = Use fast tokenizer if available\n'
             'slow = Use slow tokenizer\n'
             'mistral = Use mistral_common tokenizer\n'
             'custom = Use specified tokenizer',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for tokenizers/models (Hugging Face)",
    )

    # Timeout configurations
    parser.add_argument('--connect-timeout', type=float, default=50.0, help='Connection timeout in seconds')
    parser.add_argument('--read-timeout', type=float, default=1000.0, help='Read timeout in seconds')
    parser.add_argument('--write-timeout', type=float, default=50.0, help='Write timeout in seconds')
    parser.add_argument('--pool-timeout', type=float, help='Connection pool timeout in seconds')

    parser.add_argument(
        "--disable-tqdm", action="store_true", help="Specify to disable tqdm progress bar.",
    )
    args = parser.parse_args()
    headers_dict = json.loads(args.headers)
    start_all_time = time.perf_counter()

    # Validate max_concurrency
    if args.max_concurrency is not None and args.max_concurrency < 0:
        raise ValueError("max_concurrency must be a non-negative integer")

    if args.test_type == 'embedding':
        asyncio.run(
            test_embedding_requests(
                api_url=args.api_url,
                headers=headers_dict,
                model=args.model,
                input_len=args.input_len,
                total_requests=args.total_requests,
                request_rate=args.request_rate,
                burstiness=args.burstiness,
                max_concurrency=args.max_concurrency,
                save_to_file=args.save_to_file,
                file_path=args.file_path,
                seed=args.seed,
                random_doc_len_enabled=args.random_doc_len_enabled,
                tokenizer=args.tokenizer,
                tokenizer_mode=args.tokenizer_mode,
                trust_remote_code=args.trust_remote_code,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                write_timeout=args.write_timeout,
                pool_timeout=args.pool_timeout,
                disable_tqdm=args.disable_tqdm
            )
        )
    elif args.test_type == 'rerank':
        asyncio.run(
            test_rerank_requests(
                api_url=args.api_url,
                headers=headers_dict,
                model=args.model,
                input_len=args.input_len,
                total_requests=args.total_requests,
                request_rate=args.request_rate,
                burstiness=args.burstiness,
                max_concurrency=args.max_concurrency,
                query_len=args.query_len,
                num_docs=args.num_docs,
                save_to_file=args.save_to_file,
                file_path=args.file_path,
                seed=args.seed,
                random_doc_len_enabled=args.random_doc_len_enabled,
                tokenizer=args.tokenizer,
                tokenizer_mode=args.tokenizer_mode,
                trust_remote_code=args.trust_remote_code,
                connect_timeout=args.connect_timeout,
                read_timeout=args.read_timeout,
                write_timeout=args.write_timeout,
                pool_timeout=args.pool_timeout,
                disable_tqdm=args.disable_tqdm
            )
        )

    end_all_time = time.perf_counter()
    print(f"Total execution time: {end_all_time - start_all_time:.2f} seconds")