# 🔥 vLLM-GCU

> **vLLM-GCU** 是由燧原科技基于原生 [vLLM](https://github.com/vllm-project/vllm) 框架适配 Enflame GCU（S60）推出的高性能推理系统，支持大语言模型（LLM）及多模态视觉语言模型（VLM）的部署与运行。该项目在保留 vLLM 核心调度策略与运行机制的基础上，针对 GCU 架构实现了高效的算子执行优化。

---

## 📌 特性一览

* 完整支持 **vLLM 0.8.0** 功能特性
* 面向燧原第三代 **S60 GCU**，深度优化推理流程
* 支持 FP16、BF16，以及 GPTQ、AWQ、INT8 等多种量化方式
* 原生支持 Qwen、LLaMa、Gemma、Mistral、ChatGLM、DeepSeek 系列 LLM（和/或VLM）推理
* 提供性能测试与批量推理工具，便于部署与评估

---

## ⚙️ 安装指南

### 🔧 系统与环境要求

* **操作系统**: Ubuntu 20.04 / 22.04
* **Python**: 3.9 \~ 3.12
* **硬件**: 燧原 S60 GCU（已部署 TopsRider 软件栈）

### 📦 安装步骤

#### 1️⃣ 安装依赖组件（Docker 环境中）

请首先参考《TopsRider 软件栈安装手册》完成基础环境部署。

#### 2️⃣ 检查依赖是否已安装

```bash
python3 -m pip list | grep -E 'vllm|xformers|tops-extension|torch-gcu|torch|torchvision'
```

若已安装，输出类似如下：

```
vllm                            0.8.0
vllm_gcu                        0.8.0+<version>
torch                           2.6.0+cpu
torch-gcu                       2.6.0+<version>
torchvision                     0.21.0+cpu
xformers                        <version>
tops-extension                  <version>
```

#### 3️⃣ 安装方式（任选其一）

✅ **方式一：使用 TopsRider 安装**

```bash
./Topsrider_xxx.run -y -C vllm-gcu
```

✅ **方式二：使用 `.whl` 包手动安装**

```bash
# 安装依赖
python3 -m pip install vllm==0.8.0
python3 -m pip install torch==2.6.0+cpu -i https://download.pytorch.org/whl/cpu
python3 -m pip install torchvision==0.21.0 -i https://download.pytorch.org/whl/cpu
python3 -m pip install torch_gcu-2.6.0+<version>*.whl
python3 -m pip install tops_extension-<version>*.whl
python3 -m pip install xformers-<version>*.whl

# 安装 vllm_gcu
python3 -m pip install vllm_gcu-0.8.0+<version>*.whl
```

---

## 🚀 使用说明

### ✅ 启动推理时必备参数

* 启动需指定：`--device=gcu`
* 仅支持 `xformers` 作为 attention backend
* 默认关闭以下功能：

  * vLLM 日志统计收集
  * Async output process 功能
  * Fork 启动模式（默认使用 `spawn`）
  * 自动输入 dump（推理失败时）
* 长序列预填充 (`chunked-prefill`) 默认关闭（>32K）
* Top-p 等后处理使用原始精度计算

---

## 🧠 模型适配指南

📚 vLLM-GCU 已支持的模型参见`vLLM-GCU 模型支持列表`，以下为Qwen2.5-32B模型推理与性能测试示例，其它模型与此类似：

#### 模型下载
*  Url: [Qwen2.5-32B-Instruct-GPTQ-Int8](https://www.modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8/files)

*  branch: `master`

*  commit id: `996af7d8`

从上述Url下载模型到`Qwen2.5-32B-Instruct-GPTQ-Int8`文件夹中。

#### 批量离线推理
```shell
python3 -m vllm_utils.benchmark_test \
 --model=[Qwen2.5-32B-Instruct-GPTQ-Int8文件夹] \
 --tensor-parallel-size=2 \
 --max-model-len=32768 \
 --output-len=128 \
 --demo=te \
 --dtype=float16 \
 --device gcu \
 --quantization=gptq
```

#### Serving模式

```shell
# 启动服务端
python3 -m vllm.entrypoints.openai.api_server \
 --model [Qwen2.5-32B-Instruct-GPTQ-Int8文件夹] \
 --tensor-parallel-size 2 \
 --max-model-len 32768 \
 --disable-log-requests \
 --block-size=64 \
 --dtype=float16 \
 --device gcu \
 --trust-remote-code


# 启动客户端
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

## 📊 性能测试与 Benchmark 工具

### 工具说明

* 离线推理：展示 GCU 并推理能力
* 性能测试：统计 TPS / TTFT / latency 等指标
* 启动方式：`vllm_utils.benchmark_test` / `benchmark_vision_language`

查看参数帮助：

```bash
python3 -m vllm_utils.benchmark_test --help
python3 -m vllm_utils.benchmark_vision_language --help
```

### 推理测试参数

| 参数名称                        | 描述                     |
| --------------------------- | ---------------------- |
| `--input-len`               | 输入 token 长度            |
| `--output-len`              | 输出 token 长度            |
| `--num-prompts`             | 请求数量                   |
| `--dtype`                   | 数据类型（float16/bfloat16） |
| `--device`                  | 固定为 `gcu`              |
| `--tensor-parallel-size`    | 并行张量数（多卡）              |
| `--quantization`            | 量化方式，如：gptq、awq、w8a16  |
| `--kv-cache-dtype`          | KV 缓存量化类型，如：int8       |
| `--quantization-param-path` | KV 量化参数文件路径            |

---

## 🧩 量化支持

### ✅ 已支持量化方法

| 方法             | 描述                                               |
| -------------- | ------------------------------------------------ |
| `GPTQ`         | 4-bit group quantization，支持 group-size 为 64 或其倍数 |
| `AWQ`          | 支持 group-size 64                                 |
| `W8A16`        | 权重量化为 INT8，激活为 FP16                              |
| `INT8 KVCache` | KV Cache 支持 INT8 精度存储（需附加配置）                     |

> ❌ 暂不支持：`g_idx` 乱序（GPTQ）、SqueezeLLM、FP8、gptq\_marlin 等

---

## 🧪 vLLM-GCU 模型支持列表

| 模型名称                   | FP16 | BF16 | W4A16 GPTQ | W8A16 GPTQ | W4A16 AWQ | W8A16 | W8A8 INT8 | INT8 KV |
| ---------------------- | ---- | ---- | ---------- | ---------- | --------- | ----- | --------- | ------- |
| **Baichuan2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **ChatGLM3**           | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **DBRX**               | ✅    | ❌    | ❌          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **DeepSeek-MoE**       | ❌    | ❌    | ❌          | ❌          | ❌         | ❌     | ✅         | ✅       |
| **DeepSeek-V3**        | ❌    | ❌    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **DeepSeek-Prover-V2** | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Gemma**              | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **codegemma**          | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **InternLM2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **LLaMA2**             | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **LLaMA3**             | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **LLaMA3.1**           | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Mixtral**            | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Qwen1.5**            | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen2**              | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen2.5**            | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen3**              | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen3-MoE**          | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **WizardCoder**        | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Yi**                 | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **gte-Qwen2**          | ✅    | ❌    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **jina-reranker-v2**   | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |

---

## 图标说明：

* ✅：已支持并验证；
* ❌：暂未支持或尚未验证；
* 空白：无明确信息或未公开测试结果；

---

## 附加说明：

1. **W4A16/W8A16 GPTQ / AWQ**：均为4bit/8bit 权重量化算法，模型需通过 Enflame TopsCompressor 工具量化；
2. **INT8（W8A8）/ INT8 KV**：需加载额外量化缓存配置文件（如 `int8_kv_cache.json`），通常适用于极限压缩下的推理部署；
3. **支持模型不断更新**，如需验证特定模型，建议联系官方获取支持清单或测试补丁；
4. **Qwen 系列支持最完备**，涵盖多个模型尺寸、量化格式和推理方式（包括视觉语言模型）；

---



## 🧠 多模态模型测试说明

`benchmark_vision_language` 提供视觉语言模型的评估能力：

* 离线推理：指定图像或视频输入
* 性能测试：伪造输入进行性能压测
* 精度评估：可扩展支持数据集评估

示例命令：

```bash
python3 -m vllm_utils.benchmark_vision_language \
 --backend vllm \
 --model [模型路径] \
 --input-vision-file [图片路径] \
 --prompt "Describe this image." \
 --device gcu \
 ...
```

---

## 🌐 Serving 模式部署

支持兼容 vLLM 的 OpenAI API 接口，可快速集成至 LangChain 等应用。

### 启动服务端：

```bash
python3 -m vllm.entrypoints.openai.api_server \
 --model=[模型路径] \
 --tensor-parallel-size=4 \
 --max-model-len=32768 \
 --gpu-memory-utilization=0.9 \
 --dtype=bfloat16 \
 --quantization-param-path=[量化路径] \
 --kv-cache-dtype=int8
```

### 启动客户端：

```bash
python3 -m vllm_utils.benchmark_serving \
 --backend=vllm \
 --dataset-name=random \
 --model=[模型路径] \
 --num-prompts=1 \
 --random-input-len=3000 \
 --random-output-len=1000
```

---

## 🧪 sampler 参数扩展（已支持）

| 参数                                                                  | 功能说明               |
| ------------------------------------------------------------------- | ------------------ |
| `--top-p`, `--top-k`                                                | Top-k / Top-p 采样控制 |
| `--presence-penalty`, `--frequency-penalty`, `--repetition-penalty` | 抑制重复性输出            |
| `--ignore-eos`                                                      | 忽略 EOS 后继续生成       |
| `--include-stop-str-in-output`                                      | 是否包含停止字符           |
| `--keep-special-tokens`                                             | 是否保留特殊 token       |
| `--strict-in-out-len`                                               | 强制固定输入/输出长度        |

---

## 📚 参考文档

* [vLLM 官方文档](https://docs.vllm.ai/en/v0.8.0/)
* [TopsRider 安装手册（联系 Enflame 获取）](https://www.enflame-tech.com/)
* [TopsCompressor 量化工具](https://egc.enflame-tech.com/)

---

## 📝 许可信息

本项目遵循 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

📧 有问题？建议提交 Issue 或联系 [support@enflame-tech.com](mailto:support@enflame-tech.com)

💡 想了解更多 Enflame GCU 能力？欢迎访问 [官网](https://www.enflame-tech.com/)
