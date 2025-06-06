## deepseek

### DeepSeek-R1
本模型推理及性能测试需要8张enflame gcu。

#### 模型下载
*  url: [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

*  branch: `main`

*  commit id: `fe1c5d0d`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1`文件夹中。

#### 使用EP+DP的并行方案部署模型

##### offline测试

* offline测试脚本，将下述代码拷贝到**offline.py**文件内
```python
import os
import argparse
from vllm import LLM, SamplingParams


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='DeepSeek-R1 inference',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='./DeepSeek-R1',
                        help='model path')
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    prompts = [
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "9.11和9.8哪个数字大",
        "strawberry中有几个r?",
        "How many r in strawberry.",
    ]
    
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32, seed=0, ignore_eos=False)
    
    llm = LLM(model=args.model,
            max_num_seqs=2,
            device='gcu',
            trust_remote_code=True,
            tensor_parallel_size=1,
            disable_log_stats=False,
            max_model_len=8192,
            dtype='bfloat16',
            enforce_eager=False,
            quantization='fp8',
            seed=0,
            gpu_memory_utilization=0.8,
            enable_expert_parallel=True,
            compilation_config={"level":3},
        )

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
* offline启动脚本，将下述代码拷贝到**offline.sh**文件内
```shell
pkill -9 python
sleep 5

valid_ip=$1
dp_size=8
DEVICE_IDS=("0" "1" "2" "3" "4" "5" "6" "7")
log_folder="./output"
mkdir -p "$log_folder"

# 获取本机IP地址和网卡名
HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
echo ${HOST_IP}
echo ${INTERFACE_NAME}

if "$HOST_IP" != "$valid_ip"; then
    echo "Error: Unsupported IP address $HOST_IP, $valid_ip"
    exit 1
fi

for i in $(seq 0 7); do
    echo rank: ${i}
    echo interface_name: ${INTERFACE_NAME}
    echo ip: ${valid_ip}
    TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
    TOPS_VISIBLE_DEVICES=${DEVICE_IDS[i]} \
    GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
    VLLM_USE_V1=0 \
    VLLM_DP_MASTER_IP=${valid_ip} \
    VLLM_DP_MASTER_PORT=54999 \
    VLLM_DP_SIZE=${dp_size} \
    VLLM_DP_RANK=${i} \
    python3.10 offline.py --model ${2} &> ${log_folder}/rank_${i}.log &
done
```
* 启动测试
```shell
bash ./offline.sh [IP] [path of DeepSeek-R1]
```
* 说明：
  * `[IP]`: 服务器的ip
  * `[patch of DeepSeek-R1]`: 模型路径

##### 性能测试

* server测试脚本，将下述代码放在**server.sh**文件内

```shell
pkill -9 python
sleep 5

valid_ip=$1
model_name=$2
max_model_len=$3
dp_size=8
DEVICE_IDS=("0" "1" "2" "3" "4" "5" "6" "7")
 
base_port=7555

log_folder="./output"
mkdir -p "$log_folder"
  
# 获取本机IP地址和网卡名
HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
echo ${HOST_IP}
echo ${INTERFACE_NAME}

if [ "$HOST_IP" != "$valid_ip" ]; then
    echo "Error: Unsupported IP address $HOST_IP, $valid_ip"
    exit 1
fi

for i in $(seq 0 7); do
    echo rank: ${i}
    echo interface_name: ${INTERFACE_NAME}
    echo ip: ${valid_ip}
 
    TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
    TOPS_VISIBLE_DEVICES=${DEVICE_IDS[i]} \
    GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
    VLLM_USE_V1=0 \
    VLLM_DP_MASTER_IP=${valid_ip} \
    VLLM_DP_MASTER_PORT=54999 \
    VLLM_DP_SIZE=${dp_size} \
    VLLM_DP_RANK=${i} \
    python3 -m vllm.entrypoints.openai.api_server \
        --host ${HOST_IP} \
        --port $((i + base_port)) \
        --model ${model_name} \
        --max-model-len=${max_model_len} \
        --block-size=64 \
        --dtype=bfloat16 \
        --trust-remote-code \
        --enable-expert-parallel \
        --gpu-memory-utilization=0.9 \
        --quantization='fp8' \
        --scheduling-policy priority \
        --num-scheduler-steps 8 \
        --max-num-seqs 32 \
        -O3 &> ${log_folder}/rank_${i}.log &
done
```
* **server**启动命令
```shell
bash ./server.sh [IP] [path of DeepSeek-R1] [max-model-len]
```
* 说明：
  * `[IP]`: 服务器的ip
  * `[patch of DeepSeek-R1]`: 模型路径
  * `[max-model-len]`: 模型支持的序列最大长度
* server启动成功后，会在**output/rank_*.log**文件内输出如下日志：
```shell
INFO:     Started server process [49281]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```
* router测试脚本，将下述代码放在**router.sh**文件内
```shell
server_ips=$1
model_name=$2
log_folder="./output"
mkdir -p "$log_folder"
server_base_port=7555

router_port=8002
router_ip=$(hostname -I | awk '{print $1}')

server_urls=" http://${server_ips}:${server_base_port} http://${server_ips}:$((server_base_port+1)) http://${server_ips}:$((server_base_port+2)) http://${server_ips}:$((server_base_port+3)) http://${server_ips}:$((server_base_port+4)) http://${server_ips}:$((server_base_port+5)) http://${server_ips}:$((server_base_port+6)) http://${server_ips}:$((server_base_port+7))"

TOPS_VISIBLE_DEVICES="" \
python3 -m vllm_utils.router \
    --server-urls ${server_urls} \
    --host ${router_ip} \
    --port ${router_port} \
    --model ${model_name} &> ${log_folder}/router.log &
```
* **router**启动命令：
```shell
bash ./router.sh [IP] [path of DeepSeek-R1]
```
* 说明：
  * `[IP]`: 服务器的ip
  * `[patch of DeepSeek-R1]`: 模型路径
* router启动成功后，会在**output/router.log**文件内输出如下日志：
```shell
Starting vLLM DP Router on http://xx.xx.xx.xx:8002
```

* client测试脚本，将下述代码放在**clien.sh**文件内

  注：需要安装以下依赖：

```shell
python3 -m pip install datasets==3.6.0
```

```shell
router_ip=$1
model_name=$2
globl_bs=$3
input_len=$4
output_len=$5
log_folder="./output"
mkdir -p "$log_folder"
router_port=8002
   
client_ip=${router_ip}
HOST_IP=$(hostname -I | awk '{print $1}')

if [ "$HOST_IP" != "$client_ip" ]; then
    echo "Host IP ${HOST_IP} does not match client IP ${client_ip}, exiting..."
    exit 1
fi

router_url="http://${router_ip}:${router_port}"
   
python3 -m vllm_utils.benchmark_serving \
    --model ${model_name} \
    --backend vllm \
    --dataset-name random \
    --num-prompts ${globl_bs} \
    --random-input-len ${input_len} \
    --random-output-len ${output_len} \
    --trust-remote-code \
    --ignore-eos \
    --strict-in-out-len \
    --keep-special-tokens \
    --base-url ${router_url} \
    --extra-body '{"priority": 1}' &> ${log_folder}/client.log &
```

* **clien**启动命令：
```shell
bash ./client.sh [IP] [path of DeepSeek-R1] [batch-size] [input-len] [output-len]
* 说明：
  * `[IP]`: 服务器的ip
  * `[patch of DeepSeek-R1]`: 模型路径
  * `[batch-size]`: 模型推理的并发数
  * `[input-len]`: 输入的token长度
  * `[output-len]`: 输出的token长度
```
* cline测试成功后，会在**output/client.log**文件内输出如下日志：
```shell
============ Serving Benchmark Result ============
Successful requests:                     xxx        
Benchmark duration (s):                  xxx  
Total input tokens:                      xxx     
Total generated tokens:                  xxx      
Request throughput (req/s):              xxx  
Output token throughput (tok/s):         xxx  
Total Token throughput (tok/s):          xxx 
---------------Time to First Token----------------
Mean TTFT (ms):                          xxx
Median TTFT (ms):                        xxx
P99 TTFT (ms):                           xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx 
Median TPOT (ms):                        xxx 
P99 TPOT (ms):                           xxx 
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P99 ITL (ms):                            xxx
==================================================
```
