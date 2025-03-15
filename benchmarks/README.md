```bash

# start router
python -m vllm_utils.router --server-urls http://0.0.0.0:8000 http://0.0.0.0:8001 --port 8002 --model /home/pretrained_models/dsr1-awq/

# start client test
python -m vllm_utils.benchmark_serving --model /home/pretrained_models/dsr1-awq/ --backend vllm --dataset-name random --num-prompts 1 --random-input-len 40 --random-output-len 100 --trust-remote-code --ignore-eos --base-url http://localhost:8002 --extra-body '{"priority": 1}'

```
