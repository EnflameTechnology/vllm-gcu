# Model Support

## vLLM-GCU Supported Models

### Text Models

| Model                   | FP16 | BF16 | W4A16 GPTQ | W8A16 GPTQ | W4A16 AWQ | W8A16 | W8A8 INT8 | INT8 KV |
| ---------------------- | ---- | ---- | ---------- | ---------- | --------- | ----- | --------- | ------- |
| **Baichuan2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **ChatGLM3**           | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **DeepSeek-V3/R1**        | ❌    | ❌    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **DeepSeek-Prover-V2** | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Gemma**              | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **codegemma**          | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **InternLM2**          | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **LLaMA(2/3/3.1)**             | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Mixtral**            | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Qwen(1.5/2/2.5/3)**            | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen3-MoE**          | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **WizardCoder**        | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Yi**                 | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **gte-Qwen2**          | ✅    | ❌    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **jina-reranker-v2**   | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **GLM4-0414**   | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅        | ✅       |
---

### Multi-modal Models
TODO

---

✅: Supported and validated

❌: Not supported or not verified

(blank): Unknown or not tested publicly
