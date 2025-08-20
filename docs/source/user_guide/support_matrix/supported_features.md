# Feature Support

vLLM-GCU aims to **stay aligned with upstream vLLM** while adding hardware-specific enhancements. The plugin actively tracks upstream features and collaborates with the community to accelerate GCU support.

Refer to the [vLLM V1 Engine User Guide](https://docs.vllm.ai/en/v0.8.1/getting_started/v1_user_guide.html) for newest upstream feature definitions. Below is the support status of **vLLM-GCU** under vLLM V0 Engine:

---

## Inference Features

| Feature                | V0 Engine     | Next Step                                       |
| ---------------------- | ------------- | ----------------------------------------------- |
| Chunked Prefill        | 🟢 Functional |                                 |
| Automatic Prefix Cache | 🟢 Functional |                      |
| Speculative Decoding   | 🟢 Functional |  Basic support                                   |
| LogProbs               | 🟢 Functional |  CI improvements planned                         |
| Prompt LogProbs        | 🟢 Functional |  CI improvements planned                         |
| Async Output           | 🟢 Functional |  CI improvements planned                         |
| Sleep Mode             | 🔴 No Plan |   |

---

## Model Adaptation

| Feature         | V0 Engine     | Next Step                                      |
| --------------- | ------------- | ---------------------------------------------- |
| LoRA            | 🟢 Functional | |
| Prompt Adapter  | 🔴 Deprecated | Deprecated by upstream                         |
| Beam Search     | 🟢 Functional | CI improvements planned                        |
| Best Of         | 🟢 Functional | |
| Guided Decoding | 🟢 Functional |  |

---

## Parallelism & Scheduling

| Feature                       | V0 Engine     |  Next Step                                                  |
| ----------------------------- | ------------- | ---------------------------------------------------------- |
| Tensor Parallel               | 🟢 Functional |  CI needed                                                  |
| Pipeline Parallel             | 🟢 Functional |  CI needed                                                  |
| Expert Parallel (MoE)         | 🟢 Functional    |  CI needed                           |
| Data Parallel                 | 🟢 Functional    |  CI improvements needed                            |
| Multi-Step Scheduler          | 🟢 Functional |  Replaced by [V1 Scheduler] |
| Prefill/Decode Disaggregation | 🟢 Functional |  1P1D supported; working on xPyD and full V1 support        |

---

## Quantization & Memory

| Feature        | V0 Engine     | Next Step                                       |
| -------------- | ------------- | ----------------------------------------------- |
| Quantization   | 🟢 Functional | W8A8/W4A16 supported; working on more methods         |
| KV Cache Dtype | 🟡 Planned | Under development |
| Graph Mode     | 🟢 Functional    | Under validation          |

---

## Multi-Modality & Architectures

| Feature         | V0 Engine     | Next Step                                          |
| --------------- | ------------- | -------------------------------------------------- |
| Multi-Modality  | 🟢 Functional | See [tutorial][multimodal]; optimizing more models |
| Pooling         | 🟡 Planned | Extending model support and adding CI              |
| Encoder-Decoder | 🟡 Planned    | Under planned                     |

---

### Legend

* 🟢 **Functional**: Fully implemented and stable.
* 🔵 **Experimental**: Prototype-level support; APIs and behavior may change.
* 🟡 **Planned**: Support planned or in progress.
* 🔴 **No Plan / Deprecated**: Not supported or deprecated upstream.
