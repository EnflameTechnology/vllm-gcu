# deepseek

## DeepSeek-R1
本模型推理及性能测试，需要16张enflame gcu及vllm-0.9.2版本。

### 模型下载
*  url: [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main)

*  branch: `main`

*  commit id: `56d4cbb`

将上述url设定的路径下的内容全部下载到`DeepSeek-R1`文件夹中。


### 使用 DP4-TP4SP4-EP16 并行方案部署模型 - 天河 V3 服务器

#### 性能测试

* server测试脚本，将下述代码放在**server.sh**文件内

```shell
set -x
pkill -9 python
sleep 5
rm -r /root/.cache/vllm/torch_compile_cache/

valid_ip=$1
dp_size=4
tp_size=4
base_port=7555
model_name=$2
max_model_len=65536
max_num_seqs=256
max_num_batched_tokens=28024

HOST_IP=$(hostname -I | awk '{print $1}')
echo ${HOST_IP}
dt=`date +'%Y%m%d%H%M'`
name="$(date +%m%d)_online"

log_folder="./logs/${name}/${dt}_server"
mkdir -p "$log_folder"

valid=false
host_index=-1
for idx in "${!valid_ips[@]}"; do
    if [[ "$HOST_IP" == "${valid_ips[idx]}" ]]; then
        valid=true
        host_index=$idx
        break
    fi
done
 
if ! $valid; then
    echo "Error: Unsupported IP address $HOST_IP"
    exit 1
fi

EFRT_ENABLE_CTX_SYNC_FOR_FREE=true \
EFRT_STREAM_SYNC_USE_POLLING=true \
TOPS_STREAM_SCHEDULE_CREDIT=4 \
PYTORCH_GCU_ALLOC_CONF=backend:topsMallocAsync \
VLLM_GCU_RANK_LOG_PATH=${log_folder} \
VLLM_USE_V1=1 \
VLLM_GCU_DEEPSEEK_FUSION=1 \
VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
PYTORCH_EFML_BASED_GCU_CHECK=1 \
TORCHGCU_INDUCTOR_ENABLE=0 \
TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
vllm serve ${model_name} \
    --host ${HOST_IP} \
    --port ${base_port} \
    --max-model-len ${max_model_len} \
    --max-seq-len-to-capture ${max_model_len} \
    --max-num-batched-tokens ${max_num_batched_tokens} \
    --no-enable-prefix-caching \
    --block-size 64 \
    --dtype bfloat16 \
    --data-parallel-size ${dp_size} \
    --tensor-parallel-size ${tp_size} \
    --trust-remote-code \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.9 \
    --compilation_config '{"full_cuda_graph":true}' \
    --max-num-seqs ${max_num_seqs} \
    --cuda-graph-sizes ${max_num_seqs} \
    --quantization 'fp8' \
    --kv-cache-dtype 'fp8' &> ${log_folder}/server.log &
```
* **server**启动命令
```shell
bash ./server.sh [IP] [path of DeepSeek-R1]
```
* 说明：
  * `[IP]`: 服务器的ip
  * `[path of DeepSeek-R1]`: 模型路径
* server启动成功后，会在`log_folder`路径下的**server.log**文件内输出如下日志：
```shell
INFO:     Started server process [xxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

* client测试脚本，将下述代码放在**client.sh**文件内

  注：需要安装以下依赖：

```shell
python3 -m pip install datasets==3.6.0
```

```shell
server_ip=$1
model_name=$2
global_bs=$3
input_len=$4
output_len=$5
num_prompts=$6

dt=`date +'%Y%m%d%H%M'`
name="$(date +%m%d)_online"

log_folder="./logs/${name}/${dt}_client"
mkdir -p "$log_folder"
server_port=7555

server_url="http://${server_ip}:${server_port}"

vllm bench serve \
    --model ${model_name} \
    --dataset-name random \
    --num-prompts ${num_prompts} \
    --max-concurrency ${global_bs} \
    --random-input-len ${input_len} \
    --random-output-len ${output_len} \
    --trust-remote-code \
    --ignore-eos \
    --base-url ${server_url} \
    --save-result \
    --save-detailed \
    --result-dir ${log_folder} \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --metric-percentiles "25,50,75,90,99,100" &> ${log_folder}/client.log &
```

* **client**启动命令：
```shell
bash ./client.sh [IP] [path of DeepSeek-R1] [batch-size] [input-len] [output-len] [num-prompts]
* 说明：
  * `[IP]`: 服务器的ip
  * `[path of DeepSeek-R1]`: 模型路径
  * `[batch-size]`: 模型推理的并发数
  * `[input-len]`: 输入的token长度
  * `[output-len]`: 输出的token长度
  * `[num-prompts]`: 本次推理一共发送的请求总数，建议设置为`[batch-size]`的2~10倍
```
* client测试成功后，会在`log_folder`路径下的**client.log**文件内输出如下日志：
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
P25 TTFT (ms):                           xxx
P50 TTFT (ms):                           xxx
P75 TTFT (ms):                           xxx
P90 TTFT (ms):                           xxx
P99 TTFT (ms):                           xxx
P100 TTFT (ms):                          xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P25 TPOT (ms):                           xxx
P50 TPOT (ms):                           xxx
P75 TPOT (ms):                           xxx
P90 TPOT (ms):                           xxx
P99 TPOT (ms):                           xxx
P100 TPOT (ms):                          xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P25 ITL (ms):                            xxx
P50 ITL (ms):                            xxx
P75 ITL (ms):                            xxx
P90 ITL (ms):                            xxx
P99 ITL (ms):                            xxx
P100 ITL (ms):                           xxx
----------------End-to-end Latency----------------
Mean E2EL (ms):                          xxx
Median E2EL (ms):                        xxx
P25 E2EL (ms):                           xxx
P50 E2EL (ms):                           xxx
P75 E2EL (ms):                           xxx
P90 E2EL (ms):                           xxx
P99 E2EL (ms):                           xxx
P100 E2EL (ms):                          xxx
==================================================
```

## DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED

本模型推理及性能测试，需要两台机器共 16 张 enflame gcu及vllm-0.9.2版本。

### 模型下载

将模型下载到`DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED`文件夹中。


### 使用 DP4-TP4SP4-EP16 并行方案部署模型 - 天河 V2 服务器

#### 性能测试

* 在主节点上执行 **server-1.sh** 脚本，其内容是：

```shell
set -x
pkill -9 python
sleep 5
rm -rf /root/.cache/vllm/torch_compile_cache/*

dp_master_ip=$1
pretrained_model=$2
server_port=8002
max_model_len=65536
kv_cache_dtype=fp8
block_size=64
dtype=bfloat16
dp_size=4
dp_size_local=2
dp_start_rank=0
dp_rpc_port=13345
tp_size=4
gpu_mem_util=0.9
max_num_seqs=42
cuda_graph_sizes="1 2 3 4 5 6 7 8 12 16 20 24 28 32 36 40 42"
quantization=w4a8
max_num_batched_tokens=28024
seed=1234

HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
echo ${HOST_IP}
echo ${INTERFACE_NAME}
dt=`date +'%Y%m%d%H%M'`
name=V1Engine-S1
log_folder="./logs/${name}/${dt}_server-1"
mkdir -p "$log_folder"

PYTORCH_GCU_ALLOC_CONF=backend:topsMallocAsync \
TORCH_ECCL_ASYNC_ERROR_HANDLING=0 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=999999999 \
VLLM_GCU_RANK_LOG_PATH='./logs/' \
ECCL_IB_HCA="^=mlx5_16,mlx5_17" \
GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
VLLM_USE_V1=1 \
VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
PYTORCH_EFML_BASED_GCU_CHECK=1 \
TORCHGCU_INDUCTOR_ENABLE=0 \
TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
VLLM_GCU_DEEPSEEK_FUSION=0 \
"${profiler_args[@]}" \
vllm serve ${pretrained_model} \
        --max-model-len ${max_model_len} \
        --max-seq-len-to-capture ${max_model_len} \
        --kv-cache-dtype ${kv_cache_dtype} \
        --block-size ${block_size} \
        --dtype ${dtype} \
        --data-parallel-size ${dp_size} \
        --data-parallel-size-local ${dp_size_local} \
        --data-parallel-start-rank ${dp_start_rank}  \
        --data-parallel-address ${dp_master_ip} \
        --data-parallel-rpc-port ${dp_rpc_port} \
        --tensor-parallel-size ${tp_size} \
        --trust-remote-code \
        --enable-expert-parallel \
        --gpu-memory-utilization ${gpu_mem_util}  \
        --compilation_config '{"full_cuda_graph":true}' \
        --max-num-seqs ${max_num_seqs}  \
        --cuda-graph-sizes ${cuda_graph_sizes} \
        --quantization ${quantization} \
        --max-num-batched-tokens ${max_num_batched_tokens} \
        --seed ${seed}} \
        --no-enable-prefix-caching \
        --port ${server_port} &> ${log_folder}/server.log &
```

* 在从节点上执行 **server-2.sh**，其内容是：

```shell
set -x
pkill -9 python
sleep 5
rm -rf /root/.cache/vllm/torch_compile_cache/*

dp_master_ip=$1
pretrained_model=$2
max_model_len=65536
kv_cache_dtype=fp8
block_size=64
dtype=bfloat16
dp_size=4
dp_size_local=2
dp_start_rank=2
dp_rpc_port=13345
tp_size=4
gpu_mem_util=0.9
max_num_seqs=42
cuda_graph_sizes="1 2 3 4 5 6 7 8 12 16 20 24 28 32 36 40 42"
quantization=w4a8
max_num_batched_tokens=28024
seed=1234

HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
echo ${HOST_IP}
echo ${INTERFACE_NAME}
dt=`date +'%Y%m%d%H%M'`
name=V1Engine-S2
log_folder="./logs/${name}/${dt}_server-2"
mkdir -p "$log_folder"

PYTORCH_GCU_ALLOC_CONF=backend:topsMallocAsync \
TORCH_ECCL_ASYNC_ERROR_HANDLING=0 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=999999999 \
VLLM_GCU_RANK_LOG_PATH='./logs/' \
ECCL_IB_HCA="^=mlx5_17,mlx5_16" \
GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
VLLM_USE_V1=1 \
VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
PYTORCH_EFML_BASED_GCU_CHECK=1 \
TORCHGCU_INDUCTOR_ENABLE=0 \
TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
VLLM_GCU_DEEPSEEK_FUSION=0 \
"${profiler_args[@]}" \
vllm serve ${pretrained_model} \
    --headless \
    --max-model-len ${max_model_len} \
    --max-seq-len-to-capture ${max_model_len} \
    --kv-cache-dtype ${kv_cache_dtype} \
    --block-size ${block_size} \
    --dtype ${dtype} \
    --data-parallel-size ${dp_size} \
    --data-parallel-size-local ${dp_size_local} \
    --data-parallel-start-rank ${dp_start_rank}  \
    --data-parallel-address ${dp_master_ip} \
    --data-parallel-rpc-port ${dp_rpc_port} \
    --tensor-parallel-size ${tp_size} \
    --trust-remote-code \
    --enable-expert-parallel \
    --gpu-memory-utilization ${gpu_mem_util}  \
    --compilation_config '{"full_cuda_graph":true}' \
    --max-num-seqs ${max_num_seqs}  \
    --cuda-graph-sizes ${cuda_graph_sizes} \
    --quantization ${quantization} \
    --seed ${seed}} \
    --no-enable-prefix-caching \
    --max-num-batched-tokens ${max_num_batched_tokens}  &> ${log_folder}/server.log &
```
* 说明：
  * 需安装 net-tools 工具
  * `ECCL_IB_HCA`: 需按每台机器实际 InfiniBand 网卡配置（可通过 ibstatus 命令查看），通过 ECCL_IB_HCA 环境变量指定或排除网卡，不同机器设置可能不同。
    * 例：如果使用 mlx5_16 和 mlx5_17 网卡，则可这样设置环境变量: `export ECCL_IB_HCA="mlx5_16,mlx5_17"`
    * 例：如果不使用 mlx5_16 和 mlx5_17 网卡，则可这样设置环境变量: `export ECCL_IB_HCA="^=mlx5_16,mlx5_17"`


* **server**启动命令
```shell
# 先在主节点上执行：
bash ./server-1.sh [dp_master_ip] [path of DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED]

# 再在从节点上执行：
bash ./server-2.sh [dp_master_ip] [path of DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED]
```
* 说明：
  * `[dp_master_ip]`: 主节点服务器的 IP
  * `[path of DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED]`: 模型路径
* server 启动成功后，会在主节点`log_folder`路径下的**server.log**文件内输出如下日志：
```shell
INFO:     Started server process [xxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

* client 测试脚本，将下述代码放在**client.sh**文件内

  注：需要安装以下依赖：

```shell
python3 -m pip install datasets==3.6.0
```

```shell
server_ip=$1
model_name=$2
global_bs=$3
input_len=$4
output_len=$5
num_prompts=$6

dt=`date +'%Y%m%d%H%M'`
name="$(date +%m%d)_online"

log_folder="./logs/${name}/${dt}_client"
mkdir -p "$log_folder"
server_port=8002

server_url="http://${server_ip}:${server_port}"

vllm bench serve \
    --model ${model_name} \
    --dataset-name random \
    --num-prompts ${num_prompts} \
    --max-concurrency ${global_bs} \
    --random-input-len ${input_len} \
    --random-output-len ${output_len} \
    --trust-remote-code \
    --ignore-eos \
    --base-url ${server_url} \
    --save-result \
    --save-detailed \
    --result-dir ${log_folder} \
    --percentile-metrics 'ttft,tpot,itl,e2el' \
    --metric-percentiles "25,50,75,90,99,100" &> ${log_folder}/client.log &
```

* **client**启动命令：
```shell
bash ./client.sh [IP] [path of DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED] [batch-size] [input-len] [output-len] [num-prompts]
* 说明：
  * `[IP]`: 主节点服务器的 IP
  * `[path of DeepSeek-R1-W4AFP8_i8_AWQ_REARRAGED]`: 模型路径
  * `[batch-size]`: 模型推理的并发数
  * `[input-len]`: 输入的token长度
  * `[output-len]`: 输出的token长度
  * `[num-prompts]`: 本次推理一共发送的请求总数，建议设置为`[batch-size]`的 2~10 倍
```
* client测试成功后，会在 client 的`log_folder`路径下的**client.log**文件内输出如下日志：
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
P25 TTFT (ms):                           xxx
P50 TTFT (ms):                           xxx
P75 TTFT (ms):                           xxx
P90 TTFT (ms):                           xxx
P99 TTFT (ms):                           xxx
P100 TTFT (ms):                          xxx
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          xxx
Median TPOT (ms):                        xxx
P25 TPOT (ms):                           xxx
P50 TPOT (ms):                           xxx
P75 TPOT (ms):                           xxx
P90 TPOT (ms):                           xxx
P99 TPOT (ms):                           xxx
P100 TPOT (ms):                          xxx
---------------Inter-token Latency----------------
Mean ITL (ms):                           xxx
Median ITL (ms):                         xxx
P25 ITL (ms):                            xxx
P50 ITL (ms):                            xxx
P75 ITL (ms):                            xxx
P90 ITL (ms):                            xxx
P99 ITL (ms):                            xxx
P100 ITL (ms):                           xxx
----------------End-to-end Latency----------------
Mean E2EL (ms):                          xxx
Median E2EL (ms):                        xxx
P25 E2EL (ms):                           xxx
P50 E2EL (ms):                           xxx
P75 E2EL (ms):                           xxx
P90 E2EL (ms):                           xxx
P99 E2EL (ms):                           xxx
P100 E2EL (ms):                          xxx
==================================================
```