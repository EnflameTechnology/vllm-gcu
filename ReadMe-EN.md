# 🔥 vLLM-GCU

> **vLLM-GCU** is an efficient inference system developed by Enflame Technology based on the original [vLLM](https://github.com/vllm-project/vllm) framework, optimized for the Enflame GCU (S60). It supports the deployment and execution of Large Language Models (LLMs) and Vision-Language Models (VLMs). While retaining the core scheduling strategies and execution mechanisms of vLLM, this project introduces operator-level optimizations tailored for the GCU architecture.

---

<p align="center">
  <a href="./ReadMe-EN.md">English</a> |
  <a href="./ReadMe.md">简体中文</a> |
</p>

## 📌 Key Features

* Fully supports **vLLM 0.11.0** capabilities
* Deeply optimized inference pipeline for **Enflame S60 GCU**
* Supports various quantization formats, including GPTQ, AWQ, INT8, in addition to FP16 and BF16.
* Native support for Qwen, LLaMa, Gemma, Mistral, ChatGLM, DeepSeek series of LLMs (and/or VLMs)
* Includes performance benchmarking and batch inference tools for deployment and evaluation

---

## ⚙️ Installation Guide

### 🔧 System Requirements

* **OS**: Ubuntu 22.04
* **Python**: 3.10 \~ 3.12 (default python version `3.10+`)
* **Hardware**: Enflame S60 GCU (with TopsRider **i3x 3.6+** software stack installed)

### 📦 Installation Steps


#### 1️⃣ Pull Enflame GCU Docker & Update Driver

Pull GGCU docker environment for vLLM-GCU v0.11.0 compilation
```bash
IMAGE=registry-egc.enflame-tech.com/artifacts/vllm_gcu:v0.11.0-TR3.7.107-ubuntu2204

docker run --name vllm-gcu -d \
  -v /home:/home \
  --shm-size 8G \
  --ipc=host --network host \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  "$IMAGE" \
  tail -f /dev/null
```

Update Host GCU Driver

```bash
# Obtain driver from the docker
docker cp vllm-gcu:/enflame/driver ./
# Update GCU driver
sudo driver/enflame-x86_64-gcc-1.7.2.2402-20260429134535.run -y
# Restart Docker for the driver update to take effect
docker restart vllm-gcu
```

#### 2️⃣ Compile and Install within Docker

Get source code
```bash
cd /home
git clone https://github.com/EnflameTechnology/vllm-gcu.git
```

Compile and install vLLM-GCU
```bash
docker exec -it vllm-gcu bash
cd vllm-gcu
# compile
python3 setup.py bdist_wheel
# install
python3 -m pip install ./dist/vllm_gcu-0.11.0*.whl
```

## 🚀 Usage Instructions

### ✅ Required Parameters for Inference

* Enable flash attention (optional) and disable Torch Inductor (must)
```
export VLLM_ATTENTION_BACKEND=FLASH_ATTN # enable flash attention
export TORCHGCU_INDUCTOR_ENABLE=0 # disable Torch Inductor
```

* Must specify: `--device=gcu`

* Support `xformers` and `flash-attn` as the attention backend

* The following features are disabled by default:

  * vLLM logging and statistics collection
  * Async output processing
  * Fork mode (`spawn` is used instead)
  * Auto input dumping on inference failure

* Chunked prefill (>32K sequences) is disabled by default

* Top-p and related post-processing are computed in native precision

---

## 🧠 Model Adaptation Guide

📚 For supported models, refer to the `vLLM-GCU Supported Models` list. Below is an example using the Qwen2.5-32B model for inference and benchmarking. The process is similar for other models.

#### Download the model

* URL: [Qwen2.5-32B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8/files)
* Branch: `master`
* Commit ID: `996af7d8`

Download to the folder named `Qwen2.5-32B-Instruct-GPTQ-Int8`.

#### Batch Offline Inference

```bash
docker exec -it vllm-gcu bash
python3 -m vllm_utils.benchmark_throughput \
 --model=[Qwen2.5-32B-Instruct-GPTQ-Int8 folder path] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### Serving Mode

```bash
docker exec -it vllm-gcu bash
# Start server
python3 -m vllm.entrypoints.openai.api_server \
 --model [Qwen2.5-32B-Instruct-GPTQ-Int8 folder path] \
 --tensor-parallel-size 2 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code

# Start client
python3 -m vllm_utils.benchmark_serving \
 --backend vllm \
 --dataset-name random \
 --model [path of Qwen2.5-32B-Instruct-GPTQ-Int8] \
 --num-prompts 1 \
 --random-input-len 1024 \
 --random-output-len 1024 \
 --trust-remote-code \
 --ignore_eos \
 --strict-in-out-len \
 --keep-special-tokens
```

---

## 📊 Performance Benchmark Tools

### Tool Functions

* **Offline Inference**: Demonstrates GCU's raw inference capability
* **Performance Metrics**: Measures TPS / TTFT / latency
* **Launch with**: `vllm_utils.benchmark_throughput`

View help:

```bash
python3 -m vllm_utils.benchmark_throughput --help
```

### Common Parameters

| Parameter                   | Description                         |
| --------------------------- | ----------------------------------- |
| `--input-len`               | Input token length                  |
| `--output-len`              | Output token length                 |
| `--num-prompts`             | Number of requests                  |
| `--dtype`                   | Data type: float16/bfloat16         |
| `--device`                  | Must be `gcu`                       |
| `--tensor-parallel-size`    | Tensor parallelism (multi-card)     |
| `--quantization`            | Quantization method: gptq/awq/w8a16 |
| `--kv-cache-dtype`          | KV cache type: int8                 |
| `--quantization-param-path` | Path to KV quant config file        |

---

## 🧩 Quantization Support

### ✅ Supported Methods

| Method         | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `GPTQ`         | 4-bit group quantization; group-size must be 64 or multiple |
| `AWQ`          | Group-size = 64 supported                                   |
| `W8A16`        | Weights INT8, activations FP16                              |
| `INT8 KVCache` | KV cache supports INT8 format (requires config)             |

> ❌ Not yet supported: `g_idx` shuffle (GPTQ), SqueezeLLM, FP8, gptq\_marlin, etc.

---

## 🧪 vLLM-GCU Supported Models


| Model                   | FP16 | BF16 | W4A16 GPTQ | W8A16 GPTQ | W4A16 AWQ | W8A16 | W8A8 INT8 | INT8 KV |
| ---------------------- | ---- | ---- | ---------- | ---------- | --------- | ----- | --------- | ------- |
| **Baichuan2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **ChatGLM3**           | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **DBRX**               | ✅    | ❌    | ❌          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **DeepSeek-V3/R1/V3.2**| ❌    | ❌    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **DeepSeek-Prover-V2** | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Gemma**              | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **codegemma**          | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **InternLM2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **LLaMA(2/3/3.1)**             | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Mixtral**            | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Qwen(1.5/2/2.5/3)**            | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen3-MoE**          | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **Qwen3-Next**         | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **GLM4**               | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **WizardCoder**        | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Yi**                 | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **gte-Qwen2**          | ✅    | ❌    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **jina-reranker-v2**   | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Step3/VL**           | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **GPT-OSS**            | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
---

✅: Supported and validated

❌: Not supported or not verified

(blank): Unknown or not tested publicly

---

## Additional Notes:

1. **W4A16/W8A16 GPTQ / AWQ**: These are 4-bit / 8-bit weight-only quantization algorithms. Models must be quantized using the Enflame TopsCompressor tool.

2. **INT8 (W8A8) / INT8 KV**: Requires loading an additional quantization configuration file (e.g., `int8_kv_cache.json`), typically used for inference deployments under extreme compression scenarios.

3. **Model support is continuously evolving**. For validation of specific models, it is recommended to contact Enflame for an official support list or test patches.

4. **The Qwen series is most comprehensively supported**, covering various model sizes, quantization formats, and inference modes (including vision-language models).

---

## 🌐 Serving Deployment

Supports OpenAI-compatible API (vLLM), can be integrated with LangChain and others.

### Start Server:

```bash
python3 -m vllm.entrypoints.openai.api_server \
 --model=[model path] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --gpu-memory-utilization=0.9 \
 --dtype=bfloat16 \
 --quantization-param-path=[quant config path] \
 --kv-cache-dtype=int8
```

### Start Client:

```bash
python3 -m vllm_utils.benchmark_serving \
 --backend=vllm \
 --dataset-name=random \
 --model=[model path] \
 --num-prompts=1 \
 --random-input-len=3000 \
 --random-output-len=1000
```

---

## 🧪 Sampler Parameters (Supported)

| Parameter                                                           | Description                        |
| ------------------------------------------------------------------- | ---------------------------------- |
| `--top-p`, `--top-k`                                                | Top-k / Top-p sampling             |
| `--presence-penalty`, `--frequency-penalty`, `--repetition-penalty` | Controls repetition in output      |
| `--ignore-eos`                                                      | Ignore EOS and continue generation |
| `--include-stop-str-in-output`                                      | Include stop tokens in output      |
| `--keep-special-tokens`                                             | Retain special tokens              |
| `--strict-in-out-len`                                               | Enforce fixed input/output lengths |

---

## 📚 References

* [vLLM Official Docs](https://docs.vllm.ai/en/v0.8.0/)
* [TopsRider Installation Guide (Contact Enflame)](https://www.enflame-tech.com/)
* [TopsCompressor Quantization Tool](https://egc.enflame-tech.com/)

---

## 📝 License

This project is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

📧 Questions? Please submit an issue or contact [support@enflame-tech.com](mailto:support@enflame-tech.com)

💡 Want to learn more about Enflame GCU? Visit [Enflame Official Website](https://www.enflame-tech.com/)
