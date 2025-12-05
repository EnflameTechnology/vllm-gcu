## Deepseek-V3.1-Terminus PD 分离 MTP1 推理

本文档说明如何按 PD 分离（Prefill/Decode）方式启动 1P1D（1 Prefill 节点 + 1 Decode 节点， 8+8卡），并使用 dp2-tp4-sp4-ep8-w8a8-c8 的参数配置启动 vLLM 服务与代理。文末附完整脚本源码，便于直接对照与复用。

### 方案与模型说明

#### PD 分离（Prefill/Decode Disaggregation）概念
- 定义：PD 分离即将 “Prefill（填充，生成 KV 缓存）” 与 “Decode（解码，逐 token 生成）” 两阶段拆分为独立实例，通过 KV 连接器实现跨实例 KV 缓存传输。
- 动机：Prefill 计算密集、Decode 内存带宽密集，性能特征差异大；若耦合会造成资源浪费与延迟恶化，需分离以分别优化。
- 优势：
  - 延迟更优：降低首 token 生成时间（TTFT）与 token 间延迟（ITL）。
  - 资源更高效：Prefill/Decode 分别利用计算与带宽资源。
  - 弹性易扩展：可按业务分别扩缩容。
  - 负载更隔离：避免 Prefill 突发影响 Decode 稳定性。

#### 模型来源
- 来源：[DeepSeek-V3.1-Terminus](https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.1-Terminus/files)
- branch: master
- commit_id: 9c9951d1
- 模型保存路径：`/home/pretrained_models/deepseek-v3.1-terminus`。

#### 主要配置
- 关键配置与含义：
  - `-dp 2 -tp 4`：数据/张量并行组合，需两端保持一致；
  - `--enable-expert-parallel`：启用 MoE 专家并行；
  - `--quantization fp8` 与 `--kv-cache-dtype fp8`：计算与 KV 缓存采用 FP8 量化，显著降低显存占用，提高吞吐；
  - `--dtype bfloat16`：保持计算稳定性与兼容性；
  - `--max_model_len 65536`：长上下文，Prefill 压力较大，适合 PD 分离；
  - `--long-prefill-token-threshold 7000`、`--max-long-partial-prefills 1`：针对超长提示分段/部分预填充策略；
  - `--speculative-config {method: deepseek_mtp, num_speculative_tokens: 1}`：开启 DeepSeek MTP 推测解码；
  - `VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER=1`：启用全量 KV 传输，确保 Prefill→Decode 的上下文无缝衔接。
- 实际影响：
  - Prefill 侧更依赖内存与带宽；Decode 侧关注低延迟与稳定吞吐，PD 分离能分别优化；
  - FP8 KV 与全量 KV 传输对互联带宽与栈配置（ECCL/UCX）更敏感；
  - MoE（ep=8）对跨卡通信质量有要求，自动网卡亲和与设备筛选能降低抖动。

### 目录结构
请将以下脚本置于同一目录中，Prefill 与 Decode 两台机器均建议保持相同目录结构（便于维护与迁移）。`logs/` 及其子目录由脚本在运行时自动创建。

```bash
<workspace-root>
└─ 1p1d-dp2-tp4sp4-ep8-c8-mtp1-w8a8-dp-sampler
   ├─ p-server.sh                # Prefill 节点启动脚本（在 Prefill 节点上执行）
   ├─ d-server.sh                # Decode 节点启动脚本（在 Decode 节点上执行）
   ├─ net-config.sh              # 公共网络设备检测/生成工具（两台机器都需要）
   ├─ proxy.sh                   # 代理启动脚本（推荐在 Prefill 节点执行，亦可独立节点）
   ├─ toy_proxy_server.py        # 代理服务实现，供 proxy.sh 调用
   └─ logs/                      # 运行后自动生成
      └─ <hostname>/
         └─ *.log                # 运行日志（prefill/decode/proxy）
```

放置与执行要点：
- Prefill 节点：需具备该目录与全部脚本；依次执行 `bash p-server.sh`，确认 Prefill 和 Decode 节点都成功后再执行 `bash proxy.sh`。
- Decode 节点：需具备该目录与脚本；执行 `bash d-server.sh`。
- 两端都需要 `net-config.sh`。

### 环境准备

请参考 [安装文档](../install/install.md) 准备好基础运行环境，再源码安装 ucx 和 nixl 以便正常运行 DeepSeek-V3.1-Terminus PD 分离。

#### 源码安装 ucx 1.19.0

```bash
# download ucx source code
wget https://github.com/openucx/ucx/archive/refs/tags/v1.19.0.tar.gz
tar -zxvf v1.19.0.tar.gz
cd ucx-1.19.0
# compile ucx
./autogen.sh
./configure                          \
    --prefix=/usr/local                \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-verbs                       \
    --with-dm                          \
    --without-go  \
    --without-java \
    --enable-mt

make -j8
make install
ldconfig
```

#### 源码安装 nixl 0.5.1

```bash
# uninstall nixl
pip uninstall nixl -y --break-system-packages


# download nixl-0.5.1 source code
https://github.com/ai-dynamo/nixl/archive/refs/tags/0.5.1.tar.gz
tar -zxvf 0.5.1.tar.gz
cd nixl-0.5.1

# install necessary packages
pip install meson --break-system-packages
pip install ninja --break-system-packages
pip install pybind11 --break-system-packages


# compile
meson setup --reconfigure build --buildtype=release -Ducx_path=/usr/local/lib/ucx/ -Dinstall_headers=true -Ddisable_gds_backend=false
cd build/ && ninja && ninja install


# install nixl
cd ../ && pip install . --break-system-packages
```

### 启动流程
在脚本所在目录内执行：

1) Prefill 节点执行：

启动前可按需在 `p-server.sh` 中设置或通过环境变量覆盖关键参数（如 `PREFILL_BIND_HOST`、`PREFILL_PORT`、网络接口相关变量）。详细说明见下文“p-server.sh（Prefill 节点）”与“关键配置说明”。

```bash
bash p-server.sh
```

成功后服务监听：`http://<PREFILL_BIND_HOST>:<PREFILL_PORT>`（见控制台打印）

2) Decode 节点执行：

启动前可按需在 `d-server.sh` 中设置或通过环境变量覆盖关键参数（如 `DECODE_BIND_HOST`、`DECODE_PORT`、网络接口相关变量）。详细说明见下文“d-server.sh（Decode 节点）”与“关键配置说明”。

```bash
bash d-server.sh
```

成功后服务监听：`http://<DECODE_BIND_HOST>:<DECODE_PORT>`（见控制台打印）

3) 启动代理（可在 Prefill/Decode/独立节点上）：

启动前请先设置代理相关环境变量（如 `PREFILL_HOSTS`、`PREFILL_PORTS`、`DECODER_HOSTS`、`DECODER_PORTS`），并与实际部署 IP/端口保持一致；详细说明见下文“toy_proxy_server.py（代理服务）”。

```bash
bash proxy.sh
```

成功后代理监听：`http://<PROXY_HOST>:<PROXY_PORT>`（按 proxy.sh 参数说明设置或使用默认）

4) Client 访问：

- 健康检查：

```bash
curl http://<PROXY_HOST>:<PROXY_PORT>/healthcheck
```

- 运行 demo：

```bash
# 模型路径需要修改为实际路径
curl "<PROXY_HOST>:<PROXY_PORT>/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
            "max_tokens": 500,
            "prompt":"李白是谁",
            "model":"/home/pretrained_models/deepseek-r",
            "stop": null,
            "stream": false
        }'
```

5) 性能压测（vLLM Bench Serve）

以下为定长随机数压测脚本，仅需放置为文件（如命名为 `vllm-bench-serve.sh`）。

```bash
export TORCHGCU_INDUCTOR_ENABLE=0
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
server_port=8192

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

调用说明：

- 参数顺序：`server_ip model_name max_concurrency input_len output_len num_prompts`
- 脚本默认访问端口 `8192`，请确保服务端使用相同端口, 或指定为 <PROXY_PORT> 端口。

示例：

```bash
# 需要更改为实际使用的模型路径
bash vllm-bench-serve.sh <PROXY_HOST> /home/pretrained_models/deepseek-v3.1-terminus 8 3500 1000 64
```

client测试成功后，会在log_folder路径下的client.log文件内输出如下日志：

```bash
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

#### 服务监听汇总（启动成功后）
- Prefill 服务：`http://<PREFILL_BIND_HOST>:<PREFILL_PORT>`（以 `p-server.sh` 控制台打印为准）
- Decode 服务：`http://<DECODE_BIND_HOST>:<DECODE_PORT>`（以 `d-server.sh` 控制台打印为准）
- Proxy 服务：`http://<PROXY_HOST>:<PROXY_PORT>`（默认 `PROXY_PORT=8192`，可通过环境变量或参数覆盖）

### 关键配置说明
 - `p-server.sh` 与 `d-server.sh` 会调用 `net-config.sh`，自动解析本机 IP 与网卡的映射关系，并据此生成以下环境变量：
  - **ECCL_SOCKET_IFNAME**: 通过 `get_interface_by_ip` 解析
  - **ECCL_IB_HCA**: 通过 `get_eccl_ib_hca_exclude_list` 生成排除列表
  - **UCX_NET_DEVICES**: 通过 `generate_ucx_net_devices` 生成设备集合
- 推理核心参数：`-dp 2 -tp 4 --enable-expert-parallel --kv-cache-dtype fp8 --quantization fp8 --max_model_len 65536 --block-size 64`。
- 代理 `toy_proxy_server.py`：先向 Prefill 发起一次非流式请求以拿到 `kv_transfer_params`，再向 Decode 以流式方式转发，接口路径为 `/v1/completions` 与 `/v1/chat/completions`。

---

### 可选特性与参数扩展（MTP / EPLB / DeepEP）
本节基于“通用 PD 分离脚本”按需追加少量参数/环境变量即可启用对应特性，无需复制整段脚本。

#### MTP（DeepSeek MTP 推测解码）
- 开关参数（已在通用脚本中存在，按需覆盖）：
  - `MTP_TOKENS`：推测 token 数，默认 `1`（即 MTP1）。
  - `--speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": '${MTP_TOKENS}'}'`
- CUDA Graph 建议：
  - Prefill：`CUDA_GRAPH_SIZES_PREFILL="1 2 3 4 5 6 7 8"` 通常足够。
  - Decode：`CUDA_GRAPH_SIZES_DECODE` 需要随 `MTP_TOKENS` 与 `--max-num-seqs` 组合设置：
    - 规则：先包含 `1..8`，再从 `8 + (MTP_TOKENS+1)` 开始，以步长 `(MTP_TOKENS+1)` 递增，直至 `--max-num-seqs * (MTP_TOKENS+1)`。
    - 例如：`MTP_TOKENS=1`、`--max-num-seqs=16` → 最大值 `32`，列表为 `1 2 3 4 5 6 7 8 10 12 ... 32`。
    - 一行生成示例（在启动前导出即可）：
      ```bash
      T=${MTP_TOKENS:-1}; N=${MAX_SEQS_DECODE:-16}
      export CUDA_GRAPH_SIZES_DECODE="1 2 3 4 5 6 7 8 $(seq $((8+T+1)) $((T+1)) $((N*(T+1))))"
      ```

#### EPLB（冗余专家负载均衡）
- 适用：MoE 模型的推理/服务侧负载均衡（与 PD 分离兼容）。
- 追加参数（建议 Decode 侧开启，Prefill 可按需保持一致）：
  - `--additional_config '{"enable_eplb":true, "num_redundant_experts":<EP_OF_DECODE>}'`
    - 其中 `num_redundant_experts` 取 PD 分离 Decode 侧的专家并行度（例如 `ep=8` → `8`）。
  - `--eplb-window-size <INT>`：统计窗口大小（与数据规模正相关，示例 `100`）。
  - `--eplb-step-interval <INT>`：更新步长（与数据规模正相关，示例 `200`）。
- 快速示例（附加到 `vllm serve` 命令末尾）：
  ```bash
  --additional_config '{"enable_eplb":true, "num_redundant_experts":8}' \
  --eplb-log-balancedness \
  --eplb-window-size 100 \
  --eplb-step-interval 200
  ```

#### DeepEP（低延迟 All2All 通信栈）
- 作用：降低 MoE All2All 延迟，改善跨卡通信时延抖动。
- 开启方式：在启动前导出环境变量（仅在 decode 节点设置， prefill 节点不设置）：
  ```bash
  # decode 节点卡数需根据实际情况设置
  num_cards_of_Decode_node=8
  export VLLM_ALL2ALL_BACKEND="deepep_low_latency"
  export VLLM_MOE_DP_CHUNK_SIZE=$((256 / num_cards_of_Decode_node))
  ```
- 与 PD 分离/EPLB/MTP 兼容，可按需组合使用。

### 源码附录

#### p-server.sh（Prefill 节点）
```bash
#!/bin/bash

pkill -9 python
pkill -9 VLLM
sleep 5
rm -rf /root/.cache/vllm/torch_compile_cache/*

hostname=`hostname`
log_folder="./logs/${hostname}/"
op_dump_dir="./logs/${hostname}/op_dump_for_qa"
vpd_folder=${log_folder}/vpd/prefill
rm -rf ${vpd_folder}
mkdir -p ${vpd_folder}
rm -rf /root/.cache/vllm/torch_compile_cache/*

timestamp2=$(date +%Y-%m%d-%H%M-%S)

mkdir -p ${log_folder}
mkdir -p ${op_dump_dir}

IP_ADDR=$(hostname -I | awk '{print $1}')
echo "[Prefill] Node IP: ${IP_ADDR} (set proxy PREFILL_HOSTS=${IP_ADDR})"

export VLLM_NIXL_SIDE_CHANNEL_HOST=${IP_ADDR}
export VLLM_NIXL_SIDE_CHANNEL_PORT=5559

# Auto-configure network devices
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/net-config.sh"

# ECCL config: derive NIC name from IP
export ECCL_SOCKET_IFNAME=$(get_interface_by_ip "${IP_ADDR}")
echo "✓ ECCL_SOCKET_IFNAME: $ECCL_SOCKET_IFNAME (IP: $IP_ADDR)"

# Generate ECCL_IB_HCA exclude list
export ECCL_IB_HCA=$(get_eccl_ib_hca_exclude_list "${ECCL_SOCKET_IFNAME}")
echo "✓ ECCL_IB_HCA: $ECCL_IB_HCA"
# UCX config: generate UCX_NET_DEVICES
export UCX_NET_DEVICES=$(generate_ucx_net_devices "${ECCL_SOCKET_IFNAME}")
echo "✓ UCX_NET_DEVICES: $UCX_NET_DEVICES"

export UCX_IB_PREFER_NEAREST_DEVICE=y
export UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y
export UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y

export VLLM_USE_V1=1
export VLLM_GCU_DEEPSEEK_FUSION=1
export VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1
# Enable full KV cache transfer
export VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER=1


export VLLM_TORCH_PROFILER_DIR="${log_folder}/profile"
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION=1
export VLLM_GCU_RANK_LOG_PATH=${log_folder}

export TOPS_STREAM_SCHEDULE_CREDIT=4

export TORCHGCU_INDUCTOR_ENABLE=0
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export TORCH_ECCL_ASYNC_ERROR_HANDLING=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export PYTORCH_GCU_ALLOC_CONF="backend:topsMallocAsync"
export VLLM_GCU_TRITON_EAGLE=0


MODEL_PATH="${1:-/home/pretrained_models/deepseek-v3.1-terminus}"
model_name=$(basename "${MODEL_PATH}")

# Tunable parameters (override via env or pass-through before script)
PREFILL_PORT=${PREFILL_PORT:-8100}
DP=${DP:-2}
TP=${TP:-4}
GPU_MEM=${GPU_MEM:-0.9}
MAX_BATCHED=${MAX_BATCHED:-8192}
MAX_LEN=${MAX_LEN:-65536}
LONG_PREFILL_TOK_TH=${LONG_PREFILL_TOK_TH:-7000}
MAX_LONG_PARTIAL=${MAX_LONG_PARTIAL:-1}
MAX_PARTIAL=${MAX_PARTIAL:-1}
BLOCK_SIZE=${BLOCK_SIZE:-64}
MTP_TOKENS=${MTP_TOKENS:-1}
MAX_SEQS_PREFILL=${MAX_SEQS_PREFILL:-4}
CUDA_GRAPH_SIZES_PREFILL=${CUDA_GRAPH_SIZES_PREFILL:-"1 2 3 4 5 6 7 8"}
SEED=${SEED:-1234}
echo "[Prefill] Listening port: ${PREFILL_PORT} (set proxy PREFILL_PORTS to include this)"



log_filename=${model_name}_server-prefill-mtp1-c8-${timestamp2}

vllm serve ${MODEL_PATH}  \
    --host ${IP_ADDR} \
    --port ${PREFILL_PORT} \
    --quantization='fp8' \
    --kv-cache-dtype='fp8' \
    --dtype=bfloat16 \
    --enable-expert-parallel \
    -dp ${DP} \
    -tp ${TP} \
    --gpu-memory-utilization ${GPU_MEM} \
    --max_num_batched_tokens ${MAX_BATCHED} \
    --max_model_len ${MAX_LEN} \
    --long-prefill-token-threshold ${LONG_PREFILL_TOK_TH} \
    --max-long-partial-prefills ${MAX_LONG_PARTIAL} \
    --max-num-partial-prefills ${MAX_PARTIAL} \
    --block-size ${BLOCK_SIZE} \
    --trust-remote-code \
    --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}' \
    --additional_config '{"async_scheduling":false, "disable_dp_sampler":false, "set_cpu_affinity": true}' \
    --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": '${MTP_TOKENS}'}' \
    --max-num-seqs ${MAX_SEQS_PREFILL} \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --cuda-graph-sizes ${CUDA_GRAPH_SIZES_PREFILL} \
    --seed ${SEED} \
    --kv-transfer-config '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}' &> ${log_folder}/${log_filename}_pd_1p1d_ep8_w8_prefill.log &


```

##### p-server.sh 参数说明与示例
- 必填/可选：
  - MODEL_PATH：第 1 个位置参数，模型路径；缺省为 `/home/pretrained_models/deepseek-v3.1-terminus`。
  - IP_ADDR：自动取本机首个非回环 IP，用于 Prefill 进程监听；无需手动设置。
- 可覆盖的环境变量（带默认值）：
  - PREFILL_PORT（默认 8100）：Prefill 服务监听端口。
  - DP（默认 2）：数据并行（Data Parallelism）度，需与 Decode 保持一致。
  - TP（默认 4）：张量并行（Tensor Parallelism）度，需与 Decode 保持一致。
  - GPU_MEM（默认 0.9）：单卡显存利用率上限（0~1）。
  - MAX_BATCHED（默认 8192）：单批可并行处理的最大 token 总数。
  - MAX_LEN（默认 65536）：模型最大上下文长度（tokens）。
  - LONG_PREFILL_TOK_TH（默认 7000）：触发长上下文分段预填充的阈值（tokens）。
  - MAX_LONG_PARTIAL（默认 1）：单请求允许的长分段预填充最大次数。
  - MAX_PARTIAL（默认 1）：单请求允许的普通部分预填充最大次数。
  - BLOCK_SIZE（默认 64）：KV 缓存块大小（tokens/块）。
  - MTP_TOKENS（默认 1）：DeepSeek MTP 推测解码的推测 token 数。
  - MAX_SEQS_PREFILL（默认 4）：Prefill 阶段同时并发序列上限。
  - CUDA_GRAPH_SIZES_PREFILL（默认 "1 2 3 4 5 6 7 8"）：记录/复用 CUDA Graph 的序列规模集合，由 max-num-seqs 和 num_speculative_tokens 决定
  - SEED（默认 1234）：随机种子。
- 一致性要求：MODEL_PATH、DP/TP 等需与 Decode 侧保持一致。
- 覆盖示例：
  ```bash
  DP=4 TP=8 PREFILL_PORT=9000 MAX_LEN=32768 \
  bash p-server.sh /mnt/models/deepseek-v3.1-terminus
  ```

#### d-server.sh（Decode 节点）
```bash
#!/bin/bash

pkill -9 python
pkill -9 VLLM
sleep 5
rm -rf /root/.cache/vllm/torch_compile_cache/*

hostname=`hostname`
log_folder="./logs/${hostname}/"
op_dump_dir="./logs/${hostname}/op_dump_for_qa"
vpd_folder=${log_folder}/vpd/decode
rm -rf ${vpd_folder}
mkdir -p ${vpd_folder}


timestamp2=$(date +%Y-%m%d-%H%M-%S)

mkdir -p ${log_folder}
mkdir -p ${op_dump_dir}

IP_ADDR=$(hostname -I | awk '{print $1}')
echo "[Decode] Node IP: ${IP_ADDR} (set proxy DECODER_HOSTS=${IP_ADDR})"

# Auto-configure network devices
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/net-config.sh"

# ECCL config: derive NIC name from IP
export ECCL_SOCKET_IFNAME=$(get_interface_by_ip "${IP_ADDR}")
echo "✓ ECCL_SOCKET_IFNAME: $ECCL_SOCKET_IFNAME (IP: $IP_ADDR)"

# Generate ECCL_IB_HCA exclude list
export ECCL_IB_HCA=$(get_eccl_ib_hca_exclude_list "${ECCL_SOCKET_IFNAME}")
echo "✓ ECCL_IB_HCA: $ECCL_IB_HCA"

# UCX config: generate UCX_NET_DEVICES
export UCX_NET_DEVICES=$(generate_ucx_net_devices "${ECCL_SOCKET_IFNAME}" false)
echo "✓ UCX_NET_DEVICES: $UCX_NET_DEVICES"


export UCX_IB_PREFER_NEAREST_DEVICE=y
export UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y
export UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y

export VLLM_USE_V1=1
export VLLM_GCU_DEEPSEEK_FUSION=1
export VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1
# Enable full KV cache transfer
export VLLM_GCU_NIXL_ENABLE_FULL_KV_TRANSFER=1


export VLLM_TORCH_PROFILER_DIR="${log_folder}/profile"
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION=1
export VLLM_GCU_RANK_LOG_PATH=${log_folder}

export TOPS_STREAM_SCHEDULE_CREDIT=4

export TORCHGCU_INDUCTOR_ENABLE=0
export TORCH_ECCL_AVOID_RECORD_STREAMS=1
export TORCH_ECCL_ASYNC_ERROR_HANDLING=0
export PYTORCH_EFML_BASED_GCU_CHECK=1
export PYTORCH_GCU_ALLOC_CONF="backend:topsMallocAsync"
export VLLM_GCU_TRITON_EAGLE=1


MODEL_PATH="${1:-/home/pretrained_models/deepseek-v3.1-terminus}"
model_name=$(basename "${MODEL_PATH}")

# Tunable parameters
DECODE_PORT=${DECODE_PORT:-8200}
DP=${DP:-2}
TP=${TP:-4}
GPU_MEM=${GPU_MEM:-0.9}
MAX_BATCHED=${MAX_BATCHED:-8192}
MAX_LEN=${MAX_LEN:-65536}
BLOCK_SIZE=${BLOCK_SIZE:-64}
MTP_TOKENS=${MTP_TOKENS:-1}
MAX_SEQS_DECODE=${MAX_SEQS_DECODE:-16}
CUDA_GRAPH_SIZES_DECODE=${CUDA_GRAPH_SIZES_DECODE:-"1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32"}
SEED=${SEED:-1234}
echo "[Decode] Listening port: ${DECODE_PORT} (set proxy DECODER_PORTS to include this)"
log_filename=${model_name}_server-decode-mtp1-c8-${timestamp2}

vllm serve ${MODEL_PATH}  \
    --host ${IP_ADDR} \
    --port ${DECODE_PORT} \
    --quantization='fp8' \
    --kv-cache-dtype='fp8' \
    --dtype=bfloat16 \
    --enable-expert-parallel \
    -dp ${DP} \
    -tp ${TP} \
    --gpu-memory-utilization ${GPU_MEM} \
    --max_num_batched_tokens ${MAX_BATCHED} \
    --max_model_len ${MAX_LEN} \
    --block-size ${BLOCK_SIZE} \
    --compilation_config '{"cudagraph_mode":"FULL_DECODE_ONLY","splitting_ops":[]}' \
    --additional_config '{"async_scheduling":true, "disable_dp_sampler":false, "set_cpu_affinity": true}' \
    --speculative-config '{"method": "deepseek_mtp", "num_speculative_tokens": '${MTP_TOKENS}'}' \
    --disable-log-requests \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --cuda-graph-sizes ${CUDA_GRAPH_SIZES_DECODE} \
    --max-num-seqs ${MAX_SEQS_DECODE} \
    --seed ${SEED} \
    --kv-transfer-config '{"kv_connector":"NixlConnector", "kv_role":"kv_both"}' &> ${log_folder}/${log_filename}_pd_1p1d_ep8_w8_decode.log &


```

##### d-server.sh 参数说明与示例
- 必填/可选：
  - MODEL_PATH：第 1 个位置参数，模型路径；缺省为 `/home/pretrained_models/deepseek-v3.1-terminus`。
  - IP_ADDR：自动取本机首个非回环 IP，用于 Decode 进程监听；无需手动设置。
- 可覆盖的环境变量（带默认值）：
  - DECODE_PORT（默认 8200）：Decode 服务监听端口。
  - DP（默认 2）：数据并行度，需与 Prefill 保持一致。
  - TP（默认 4）：张量并行度，需与 Prefill 保持一致。
  - GPU_MEM（默认 0.9）：单卡显存利用率上限（0~1）。
  - MAX_BATCHED（默认 8192）：单批可并行处理的最大 token 总数。
  - MAX_LEN（默认 65536）：模型最大上下文长度（tokens）。
  - BLOCK_SIZE（默认 64）：KV 缓存块大小（tokens/块）。
  - MTP_TOKENS（默认 1）：DeepSeek MTP 推测解码的推测 token 数。
  - MAX_SEQS_DECODE（默认 16）：Decode 阶段同时并发序列上限。
  - CUDA_GRAPH_SIZES_DECODE（默认 "1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32"）：记录/复用 CUDA Graph 的序列规模集合，由 max-num-seqs 和 num_speculative_tokens 决定
  - SEED（默认 1234）：随机种子。
- 一致性要求：MODEL_PATH、DP/TP 等需与 Prefill 侧保持一致。
- 覆盖示例：
  ```bash
  MAX_SEQS_DECODE=32 CUDA_GRAPH_SIZES_DECODE="1 2 4 8 16 32" \
  bash d-server.sh /mnt/models/deepseek-v3.1-terminus
  ```

#### net-config.sh（网络设备配置工具函数库）
```bash
#!/bin/bash
# ========================================
# Network device configuration helper functions
# ========================================
# Generate ECCL_SOCKET_IFNAME, ECCL_IB_HCA, and UCX_NET_DEVICES automatically

# Function: Get network interface name from IP address
# Args: $1 - IP address
# Output: Interface name (e.g., ens12f1np1)
get_interface_by_ip() {
    local ip="$1"
    if [ -z "$ip" ]; then
        echo ""
        return 1
    fi

    local ifname=$(ifconfig -a | grep "$ip" -B 1 | head -1 | awk -F: '{print $1}')
    echo "$ifname"
}


# Function: Build ECCL IB HCA exclude list
# Args: $1 - interface name (optional; auto-detect by default)
# Output: string like "^=mlx5_8,mlx5_9"
get_eccl_ib_hca_exclude_list() {
    local ifname="${1:-}"

    # Auto-detect primary interface if not specified
    if [ -z "$ifname" ]; then
        local host_ip=$(hostname -I | awk '{print $1}')
        ifname=$(ifconfig -a | grep -B1 "inet ${host_ip}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
    fi

    # Query IB devices
    local ibdev_output=$(ibdev2netdev 2>/dev/null || echo "")
    if [ -z "$ibdev_output" ]; then
        echo ""
        return 0
    fi

    local exclude_devices=()

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local device=$(echo "$line" | awk '{print $1}')
        local interface=$(echo "$line" | awk '{print $5}')
        local status=$(echo "$line" | awk -F"[()]" '{print $2}')

        # Only handle mlx5_* devices
        [[ $device =~ ^mlx5_[0-9]+$ ]] || continue

        # Exclude devices bound to the given interface or with Down status
        if [[ $interface == "$ifname" || $status == "Down" ]]; then
            exclude_devices+=("$device")
        fi
    done <<< "$ibdev_output"

    if [ ${#exclude_devices[@]} -gt 0 ]; then
        local IFS=","
        echo "^=${exclude_devices[*]}"
    else
        echo ""
    fi
}

# Function: Generate UCX_NET_DEVICES list
# Args: $1 - interface name (optional)
#       $2 - include-only mode (optional; default: exclude the given iface)
# Output: like "mlx5_bond_0:1,mlx5_bond_1:1,..."
generate_ucx_net_devices() {
    local ifname="${1:-}"
    local include_mode="${2:-false}"

    # Auto-detect primary interface if not specified
    if [ -z "$ifname" ]; then
        local host_ip=$(hostname -I | awk '{print $1}')
        ifname=$(ifconfig -a | grep -B1 "inet ${host_ip}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)
    fi

    # Query IB devices
    local ibdev_output=$(ibdev2netdev 2>/dev/null || echo "")
    if [ -z "$ibdev_output" ]; then
        echo ""
        return 0
    fi

    local devices=()

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        local device=$(echo "$line" | awk '{print $1}')
        local interface=$(echo "$line" | awk '{print $5}')
        local status=$(echo "$line" | awk -F"[()]" '{print $2}')

        # Only handle mlx5* devices
        [[ $device =~ ^mlx5.*_[0-9]+$ ]] || continue

        # Keep devices with Up status only
        [[ $status != "Up" ]] && continue

        # Select devices per include/exclude mode
        if [ "$include_mode" = "true" ]; then
            # Include mode: only devices on the given interface
            [[ $interface == "$ifname" ]] && devices+=("${device}:1")
        else
            # Exclude mode: skip devices on the given interface
            [[ $interface != "$ifname" ]] && devices+=("${device}:1")
        fi
    done <<< "$ibdev_output"

    if [ ${#devices[@]} -gt 0 ]; then
        local IFS=","
        echo "${devices[*]}"
    else
        echo ""
    fi
}
```

#### proxy.sh（代理启动脚本）
```bash
log_folder="./logs"
mkdir -p $log_folder
dt=`date +'%Y%m%d_%H%M'`

export VLLM_USE_V1=1
# export VLLM_LOGGING_LEVEL=DEBUG

# Resolve hosts/ports from env with sensible defaults
HOST_IP=$(hostname -I | awk '{print $1}')
PROXY_HOST=${PROXY_HOST:-${HOST_IP}}
PROXY_PORT=${PROXY_PORT:-8192}

PREFILL_HOSTS=${PREFILL_HOSTS:-${HOST_IP}}
PREFILL_PORTS=${PREFILL_PORTS:-${PREFILL_PORT:-8100}}

# Decoder is usually on a different node: require override when absent
DECODER_HOSTS=${DECODER_HOSTS:-${DECODER_HOST}}
DECODER_PORTS=${DECODER_PORTS:-${DECODE_PORT:-8200}}

MODEL_PATH=${MODEL_PATH:-/home/pretrained_models/deepseek-v3.1-terminus}

if [ -z "${DECODER_HOSTS}" ]; then
  echo "[proxy] Please set DECODER_HOSTS (e.g. export DECODER_HOSTS=10.12.xx.yy)"
  exit 1
fi

python3 toy_proxy_server.py \
--host ${PROXY_HOST} \
--port ${PROXY_PORT} \
--prefiller-hosts ${PREFILL_HOSTS} \
--prefiller-ports ${PREFILL_PORTS} \
--decoder-hosts ${DECODER_HOSTS} \
--decoder-ports ${DECODER_PORTS} \
--model ${MODEL_PATH} &> ${log_folder}/${dt}_pd_1p1d_ep8_w8_proxy.log &
```

##### proxy.sh 参数说明与示例
- 环境变量（可按需覆盖，未设置则使用默认）：
  - PROXY_HOST：代理监听地址，默认本机 IP。
  - PROXY_PORT：代理监听端口，默认 8192。
  - PREFILL_HOSTS：Prefill 节点 IP（可空格分隔多个），默认本机 IP。
  - PREFILL_PORTS：Prefill 端口（可空格分隔多个），默认取 `PREFILL_PORT` 或 8100。
  - DECODER_HOSTS：Decode 节点 IP（可空格分隔多个）。必填，如未设置将提示并退出。
  - DECODER_PORTS：Decode 端口（可空格分隔多个），默认取 `DECODE_PORT` 或 8200。
  - MODEL_PATH：模型路径，默认 `/home/pretrained_models/deepseek-v3.1-terminus`。

- 与 server 端一致性要求：
  - `PREFILL_PORTS` 必须与 Prefill 侧 `PREFILL_PORT` 一致。
  - `DECODER_PORTS` 必须与 Decode 侧 `DECODE_PORT` 一致。
  - 如配置多实例，`PREFILL_HOSTS` 与 `PREFILL_PORTS` 数量需一致；`DECODER_HOSTS` 与 `DECODER_PORTS` 数量需一致。

- 快速示例：
  - 在 Prefill 节点上启动代理，Prefill 本机，Decode 为 <DECODE_BIND_HOST>：
    ```bash
    export DECODER_HOSTS=<DECODE_BIND_HOST>
    export PREFILL_PORT=8100
    export DECODE_PORT=8200
    export PROXY_PORT=8192
    bash proxy.sh
    ```
  - 在 Decode 节点上启动代理，指向远端 Prefill：
    ```bash
    export PREFILL_HOSTS=<PREFILL_BIND_HOST>
    export DECODER_HOSTS=$(hostname -I | awk '{print $1}')
    bash proxy.sh
    ```

#### toy_proxy_server.py（代理服务）
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import os
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from vllm.logger import init_logger

logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    limits = httpx.Limits(max_connections=256)

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}/v1'
        app.state.prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=prefiller_base_url, limits=limits),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}/v1'
        app.state.decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=decoder_base_url, limits=limits),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # Model path argument
    parser.add_argument("--model",
                        type=str,
                        default="/home/pretrained_models/deepseek-v3.1-terminus",
                        help="Model path (default: /home/pretrained_models/deepseek-v3.1-terminus)")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8100])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[8200])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_service(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_service_response(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')

        # Send request to prefill service
        req_data["model"] = global_args.model
        response = await send_request_to_service(prefill_client_info, api,
                                                 req_data, request_id)

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
            req_data["model"] = global_args.model
            response_json["model"] = req_data["model"]

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            try:
                async for chunk in stream_service_response(decode_client_info,
                                                           api,
                                                           req_data,
                                                           request_id=request_id):
                    yield chunk
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError) as e:
                logger.warning(f"upstream stream aborted early: {e}")
                return

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
```
