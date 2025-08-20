# Installation

This guide explains how to install and set up **vLLM-GCU** manually or via Docker for use on Enflame GCU hardware (S60).

## Requirements

* **OS**: Ubuntu 20.04 / 22.04
* **Python**: 3.10 \~ 3.12
* **Hardware**: Enflame GCU (e.g., S60)
* **Software Stack**:

  | Software  | Required Version            | Notes                                     |
  | --------- | --------------------------- | ----------------------------------------- |
  | TopsRider_i3x | ≥ `3.4`                 | Required for GCU driver/runtime           |
  | torch-gcu | Compatible with PyTorch 2.6 | Installed via `.whl`, provided by Enflame |
  | vllm-gcu  | Based on `vLLM 0.8.0`       | Built for the vLLM GCU backend          |

---

## Environment Setup

Before installing, ensure that GCU drivers and the **TopsRider** stack are installed properly. Follow the [TopsRider Installation Manual](https://support.enflame-tech.com/onlinedoc_dev_3.4/2-install/sw_install/content/source/installation.html).

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

### Option 1: Install via TopsRider (Recommended)

> This method installs all required packages, extensions, and runtime libraries for Enflame GCU.

```bash
# Install Triton (required)
python3 -m pip install triton==3.2

# Install TopsRider and setup vLLM-GCU
chmod +x ./TopsRider_i3x_3.4.xxx.run
sudo ./TopsRider_i3x_3.4.xxx.run -y -C vllm-gcu
```

---

### Option 2: Manual Build from Source

#### Step 1: Install Required Python Packages

```bash
python3 -m pip install vllm==0.8.0
python3 -m pip install triton==3.2
python3 -m pip install torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu
python3 -m pip install torchvision==0.21.0 -i https://download.pytorch.org/whl/cpu

# Install GCU-specific extensions
python3 -m pip install torch_gcu-2.6.0+<version>.whl
python3 -m pip install tops_extension-<version>.whl
python3 -m pip install xformers-<version>.whl
```

#### Step 2: Build and Install vLLM-GCU from source code

```bash
git clone https://github.com/enflame-tech/vllm-gcu.git
cd vllm-gcu
python3 setup.py bdist_wheel
python3 -m pip install ./dist/vllm_gcu-0.8.0+<version>.whl
```

> ⚠️ Replace `<version>` with the appropriate version string.

---

## Docker Support

If you prefer containerized deployment, use the prebuilt Docker image provided by Enflame.

```bash
export IMAGE=artifact.enflame.cn/enflame_docker_release/amd64_ubuntu2004_tr:latest

# start the docker
docker run --ipc=host --network host --privileged -v /dev:/dev -d -i --name gcudocker $IMAGE

# copy driver package and install driver on the host
docker cp gcudocker:/enflame/driver ~/
cd ~/driver
sudo ./enflame-x86_64*.run

# exec the docker
docker exec -it gcudocker bash
```

The default working directory will contain vLLM and vLLM-GCU in development mode.

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
