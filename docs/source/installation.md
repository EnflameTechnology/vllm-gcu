# Installation

This guide explains how to install and set up **vLLM-GCU** manually or via Docker for use on Enflame GCU hardware (S60).

## Requirements

* **OS**: Ubuntu 22.04
* **Python**: 3.10 \~ 3.12
* **Hardware**: Enflame GCU (e.g., S60)
* **Software Stack**:

  | Software  | Required Version            | Notes                                     |
  | --------- | --------------------------- | ----------------------------------------- |
  | TopsRider_i3x | ≥ `3.6`                 | Required for GCU driver/runtime           |
  | torch-gcu | Compatible with PyTorch 2.8 | Installed via `.whl`, provided by Enflame |
  | vllm-gcu  | Based on `vLLM 0.11.0`       | Built for the vLLM GCU backend          |

---

## Environment Setup

Before installing, ensure that GCU drivers and the **TopsRider** stack are installed properly. Follow the [TopsRider Installation Manual](https://support.enflame-tech.com/onlinedoc_dev_3.6/2-install/sw_install/content/source/installation.html#id6).

### Validate GCU Installation

Run the following command to check GCU availability:

```bash
efsmi
```

Expected output includes driver version, hardware type, temperature, power, memory and GCU core usage.

```
------------------------------------------------------------------------------
-------------------- Enflame System Management Interface ---------------------
--------- Enflame Tech, All Rights Reserved. 2024-2025 Copyright (C) ---------
------------------------------------------------------------------------------
                                                                              
+2025-07-21, 03:50:09 UTC----------------------------------------------------+
| EFSMI    1.4.0.505       Driver Ver: 1.4.4.501                             |
|----------------------------------------------------------------------------|
|----------------------------------------------------------------------------|
| DEV    NAME                 | FW VER           | BUS-ID      ECC           |
| TEMP   Dpm   Pwr(Usage/Cap) | Mem     GCU Virt | DUsed       SN            |
|----------------------------------------------------------------------------|
| 0      Enflame S60          | 33.6.5           | 00:3d:00.0  Enable        |
| -      Sleep    -           | 49120MiB Disable | 0%          xxx |
+----------------------------------------------------------------------------+
|----------------------------------------------------------------------------|
| 1      Enflame S60          | 33.6.5           | 00:3e:00.0  Disable       |
| -      Sleep    -           | 49120MiB Disable | 0%          xxx |
+----------------------------------------------------------------------------+
```

---

## Installation Options

You can install vLLM-GCU in **two ways**:

* **Option 1**: Use **TopsRider installer** with prebuilt `.whl`
* **Option 2**: Build vLLM-GCU from **source code**

Choose one of the following methods depending on your usage and environment.

**Python3.10+:** 

Before installation, make sure you have python3.10+ installed and the default python version is 3.10+

```bash
# check default python version 
python3 --version

# install python3.10 if default python version < 3.10
sudo apt update && sudo apt install python3.10 -y

# switch default python to version 3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
sudo update-alternatives --config python3

# install pip for python3.10
sudo apt update && sudo apt install python3.10-distutils -y
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3

# install setuptools
python3 -m pip install setuptools
```

---

### Option 1: Install via TopsRider

> This method installs all required packages, extensions, and runtime libraries for Enflame GCU.

```bash
# Install Triton (required)
python3 -m pip install triton==3.3

# Install TopsRider and setup vLLM-GCU
chmod +x ./TopsRider_i3x_3.6.xxx.run
sudo ./TopsRider_i3x_3.6.xxx.run -y -C vllm-gcu
```

---

### Option 2: Build source code with Docker

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
# compile
python3 setup.py bdist_wheel
# install
python3 -m pip install ./dist/vllm_gcu-0.11.0*.whl
```

---

## Post-Installation Check

To verify that everything is working, run a simple script named `example.py` which contains:

```python
from vllm import LLM, SamplingParams

prompts = ["Talk about China in 100 words.", "The best country for travelling is", "Qwen3 is developed by"]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", device="gcu")

outputs = llm.generate(prompts, sampling_params)
for out in outputs:
    print(f"> {out.prompt.strip()} -> {out.outputs[0].text.strip()}")
```

Run with:

```bash
python3 example.py
```

---

## Notes

* Enflame recommends using **GCU quantized models** generated via [**TopsCompressor**](user_guide/feature_guide/quantization.md).
* If encountering compile errors with `xformers`, ensure GCC >= 9 and C++17 support.
* For full quantization support (AWQ, GPTQ, etc.), confirm that the quantization config files are in place.
