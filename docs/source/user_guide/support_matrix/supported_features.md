# Feature Support

vLLM-GCU aims to **stay aligned with upstream vLLM** while adding hardware-specific enhancements. The plugin actively tracks upstream features and collaborates with the community to accelerate GCU support.

Refer to the [vLLM V1 Engine User Guide][v1_user_guide] for upstream feature definitions. Below is the support status for **vLLM-GCU**:

---

## Inference Features

| Feature                | V0 Engine     | V1 Engine     | Next Step                                       |
| ---------------------- | ------------- | ------------- | ----------------------------------------------- |
| Chunked Prefill        | 🟢 Functional | 🟢 Functional |                                |
| Automatic Prefix Cache | 🟢 Functional | 🟢 Functional |                      |
| Speculative Decoding   | 🟢 Functional | 🟢 Functional | Basic support                                   |
| LogProbs               | 🟢 Functional | 🟢 Functional | CI improvements planned                         |
| Prompt LogProbs        | 🟢 Functional | 🟢 Functional | CI improvements planned                         |
| Async Output           | 🟢 Functional | 🟢 Functional | CI improvements planned                         |
| Sleep Mode             | 🟢 Functional | 🟢 Functional | Level=1 supported, V1 optimizations in progress |

---

## Model Adaptation

| Feature         | V0 Engine     | V1 Engine     | Next Step                                      |
| --------------- | ------------- | ------------- | ---------------------------------------------- |
| LoRA            | 🟢 Functional | 🟢 Functional |  |
| Prompt Adapter  | 🔴 Deprecated | 🔴 Deprecated | Deprecated by upstream                         |
| Beam Search     | 🟢 Functional | 🟢 Functional | CI improvements planned                        |
| Best Of         | 🟢 Functional | 🔴 Deprecated |     |
| Guided Decoding | 🟢 Functional | 🟢 Functional |                   |

---

## Parallelism & Scheduling

| Feature                       | V0 Engine     | V1 Engine     | Next Step                                                  |
| ----------------------------- | ------------- | ------------- | ---------------------------------------------------------- |
| Tensor Parallel               | 🟢 Functional | 🟢 Functional | CI needed                                                  |
| Pipeline Parallel             | 🟢 Functional | 🟢 Functional | CI needed                                                  |
| Expert Parallel (MoE)         | 🔴 No Plan    | 🟢 Functional | Functional in V1 only, CI needed                           |
| Data Parallel                 | 🔴 No Plan    | 🟢 Functional | V1 only, CI improvements needed                            |
| Multi-Step Scheduler          | 🟢 Functional | 🔴 Deprecated | Replaced by [V1 Scheduler][v1_scheduler] ([#8779][v1_rfc]) |
| Prefill/Decode Disaggregation | 🟢 Functional | 🟢 Functional | 1P1D supported; working on xPyD and full V1 support        |

---

## Quantization & Memory

| Feature        | V0 Engine     | V1 Engine       | Next Step                                       |
| -------------- | ------------- | --------------- | ----------------------------------------------- |
| Quantization   | 🟢 Functional | 🟢 Functional   | W8A8 supported; working on more methods         |
| KV Cache Dtype | 🟢 Functional | 🟢 Functional   | Set via `kv_cache_dtype` in `additional_config` |
| Graph Mode     | 🔴 No Plan    | 🔵 Experimental | Under validation ([#767][graph_mode])           |

---

## Multi-Modality & Architectures

| Feature         | V0 Engine     | V1 Engine     | Next Step                                          |
| --------------- | ------------- | ------------- | -------------------------------------------------- |
| Multi-Modality  | 🟢 Functional | 🟢 Functional | See [tutorial][multimodal]; optimizing more models |
| Pooling         | 🟢 Functional | 🟡 Planned    | Extending model support and adding CI              |
| Encoder-Decoder | 🔴 No Plan    | 🟡 Planned    | Support planned by 2025-06-30                      |

---

### Legend

* 🟢 **Functional**: Fully implemented and stable.
* 🔵 **Experimental**: Prototype-level support; APIs and behavior may change.
* 🟡 **Planned**: Support planned or in progress.
* 🔴 **No Plan / Deprecated**: Not supported or deprecated upstream.
