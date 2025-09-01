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
| **Qwen(1.5/2/2.5/3)**  | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **Qwen3-MoE**          | ✅    | ✅    | ❌          | ❌          | ✅         | ❌     | ❌         | ❌       |
| **WizardCoder**        | ✅    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **Yi**                 | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅         | ✅       |
| **gte-Qwen2**          | ✅    | ❌    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **jina-reranker-v2**   | ❌    | ✅    | ❌          | ❌          | ❌         | ❌     | ❌         | ❌       |
| **GLM4-0414**   | ✅    | ✅    | ✅          | ✅          | ✅         | ✅     | ✅        | ✅       |
---

### Multi-modal Models
| Model                   | FP16 | BF16 | W4A16 GPTQ | W8A16 GPTQ | 
| ---------------------- | ---- | ---- | ---------- | ---------- |
|  **LLaVa**            | ✅               | ❌               | ❌              | ❌
|  **GLM-4V**           | ❌               | ✅               | ❌                | ❌
|  **DeepSeek-VL2**     | ❌               | ✅               | ❌                | ❌
|   **InternVL2**        | ❌               | ✅               | ❌               | ❌
|    **InternVL2.5**      | ❌               | ✅               | ❌               | ❌
|    **LLaVa-Next**       | ✅               | ❌               | ❌               | ❌
|    **MiniCPM-V**        | ✅               | ❌               | ❌               | ❌
|    **Phi-3-vision**     | ❌               | ✅              | ❌               | ❌
|    **Qwen-VL**          | ❌               | ✅              | ❌               | ❌
|    **Qwen2-VL**         | ✅               | ❌             | ✅                | ✅
|    **Qwen2.5-VL**       | ❌               | ✅               | ❌               | ❌
|    **Yi-VL**           | ❌               | ✅               | ❌               | ❌
|    **GOT-OCR-2.0**      | ❌               | ✅               | ❌               | ❌
|    **QVQ-72B-preview**  | ❌               | ✅               | ❌               | ❌


---

✅: Supported and validated

❌: Not supported or not verified
