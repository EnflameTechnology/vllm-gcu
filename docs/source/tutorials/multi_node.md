# Multi-Node Inference (DeepSeek-R1)

## Overview

This document provides a professional guide for deploying DeepSeek-R1 in an efficient multi-node setup using Data Parallelism (DP), Tensor Parallelism (TP), and Expert Parallelism (EP) on GCU hardware. The setup targets 4 nodes (each with 8 GCUs), for a total of 32 GCUs.

### Parallelism Strategy

DeepSeek-R1 adopts a hybrid parallelism strategy for optimal performance:

* **Data Parallelism (DP)**: Each replica of the model processes a separate batch on a different node or group of devices. This maximizes throughput and allows large-scale deployment.
* **Tensor Parallelism (TP)**: Model weights are split across multiple devices to allow the inference of large models that cannot fit on a single GCU.
* **Expert Parallelism (EP)**: MoE (Mixture-of-Experts) layers are distributed among devices. This is ideal for DeepSeek’s architecture which relies heavily on MoE layers.

Combined (DP \* TP), each DP rank can manage multiple TP workers. Expert layers require all DP ranks to synchronize at each step to ensure correct behavior. For this reason, even when some ranks have no requests, dummy forward passes must be executed to maintain synchronization.

## 1. Cluster Environment Setup

### Physical Requirements

* 4 physical machines connected to the same high-speed InfiniBand (IB) network.
* All nodes should have the GCU hardware connected and powered on.
* All required driver and software packages should be placed under a consistent directory on all machines, like `/data/test/`

### Network Configuration

Each node must be accessible via its IB IP address. Ensure the following machines are accessible:

* Node 0: 192.168.1.2
* Node 1: 192.168.1.3
* Node 2: 192.168.1.4
* Node 3: 192.168.1.5

SSH access should be performed via the IB interface.

## 2. Software and Driver Installation

### Driver Installation (on each machine)

```bash
sudo rmmod enflame-peer-mem
sudo rmmod enflame
sudo bash ./TopsPlatform_*_deb_amd64.run --no-auto-load --peermem -y
```

### Docker Image Preparation

Use the pre-saved Docker image (since `docker pull` may fail):

```bash
docker load -i ubuntu2204_llm.tar
```

### Launch Docker Container

```bash
docker run -it --name test_env \
  -v /data/test:/home/workspace \
  -e ENFLAME_VISIBLE_DEVICES=all \
  -e TZ=Asia/Shanghai \
  --ipc=host -u root \
  -e ENFLAME_UMD_FLAGS="enable_gcu_coredump=true" \
  --ulimit core=-1 --security-opt seccomp=unconfined \
  --network host -v /sys/kernel:/sys/kernel \
  --privileged artifact.enflame.cn/enflame_docker_release/amd64_ubuntu2204_llm:3.4.203
```

### Python Environment Setup

```bash
python3 -m pip install datasets==3.2.0
python3 -m pip install opencv-python==4.10.0.84 opencv-python-headless==4.10.0.84
python3 -m pip install triton==3.2.0
```

### ECCL Communication Verification

Install MPI and run ECCL test:

```bash
apt install lam-runtime mpich openmpi-bin slurm-wlm-torque
cd /usr/local/bin
export ECCL_ALLTOALLV_MAXSIZE=134217728
mpirun -np 8 eccl_alltoall_perf -b 1k -e 128M -f 2
```

## 3. Multi-Node Inference Deployment

### 3.1 Start Inference Servers (on each node)

Each machine will spawn two processes, using a total of 8 devices per node.

Create a script `server.sh` and execute it on all four nodes:

```bash
#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
valid_ips=("192.168.1.2" "192.168.1.3" "192.168.1.4" "192.168.1.5")
base_port=7555
dp_size=$((${#valid_ips[@]} * 2))
VISIBLE_DEVICES=("0,1,2,3" "4,5,6,7")
max_num_batched_tokens=2048

HOST_IP=$(hostname -I | awk '{print $1}')
INTERFACE_NAME=$(ifconfig -a | grep -B1 "inet ${HOST_IP}[^0-9]" | head -n 1 | awk '{print $1}' | cut -d: -f1)

log_folder="./logs/server"
mkdir -p "$log_folder"

# Determine local rank based on IP
host_index=-1
for idx in "${!valid_ips[@]}"; do
    if [[ "$HOST_IP" == "${valid_ips[idx]}" ]]; then
        host_index=$idx
        break
    fi
done

if [[ $host_index -eq -1 ]]; then
    echo "Unsupported IP: $HOST_IP"
    exit 1
fi

for i in {0..1}; do
    rank=$((host_index * 2 + i))

    ECCL_ALLTOALLV_MAXSIZE=$((max_num_batched_tokens * dp_size *(7168+16+16)*2)) \
    TORCH_ECCL_AVOID_RECORD_STREAMS=1 \
    GLOO_SOCKET_IFNAME=${INTERFACE_NAME} \
    TOPS_VISIBLE_DEVICES=${VISIBLE_DEVICES[i]} \
    VLLM_GCU_ENABLE_SEQUENCE_PARALLEL=1 \
    VLLM_USE_V1=0 \
    VLLM_DP_MASTER_IP=${valid_ips[0]} \
    VLLM_DP_MASTER_PORT=54999 \
    VLLM_DP_SIZE=${dp_size} \
    VLLM_DP_RANK=${rank} \
    python3 -m vllm.entrypoints.openai.api_server \
        --host ${HOST_IP} \
        --port $((i + base_port)) \
        --model=/home/workspace/models/DeepSeek-R1-awq \
        --dtype bfloat16 \
        --trust-remote-code \
        --quantization moe_wna16_gcu \
        --max-model-len=${max_num_batched_tokens} \
        --tensor-parallel-size=4 \
        --gpu-memory-utilization=0.8 \
        --block-size=64 \
        --enable-expert-parallel \
        --num-scheduler-steps 8 \
        --scheduling-policy priority &> ${log_folder}/server_rank${rank}.log &
done
```

### 3.2 Start Router (only on the master node)

```bash
#!/bin/bash

server_ips=("192.168.1.2" "192.168.1.3" "192.168.1.4" "192.168.1.5")
server_base_port=7555
router_ip="192.168.1.2"
router_port=8002

HOST_IP=$(hostname -I | awk '{print $1}')
if [ "$HOST_IP" != "$router_ip" ]; then
    echo "Not the router node, exiting..."
    exit 1
fi

log_folder="./logs/router"
mkdir -p "$log_folder"

server_urls=""
for ip in "${server_ips[@]}"; do
    server_urls+=" http://${ip}:${server_base_port} http://${ip}:$((server_base_port+1))"
done

python3 -m vllm_utils.router \
    --server-urls ${server_urls} \
    --host ${router_ip} \
    --port ${router_port} \
    --model /home/workspace/models/DeepSeek-R1-awq &> ${log_folder}/router.log &
```

### 3.3 Client Request Example

Once the router is running, inference requests can be sent:

```bash
curl http://192.168.1.2:8002/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/workspace/models/DeepSeek-R1-awq",
        "prompt": "Which one is larger? 3.8 or 3.11",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

## 4. Additional Notes

* ECCL communication should be verified with `eccl_alltoall_perf`.
* `server.sh`, `router.sh`, and `client.sh` scripts must have consistent configurations for IPs and ports.
* Use `psmisc` package (`apt-get install psmisc`) for debugging processes.
* In case of driver installation failures:

  * If SSM service crashes: `sudo efsmt -d dtu.* -pcie reset hot`
  * If pip is missing: `sudo apt-get install python3-pip`