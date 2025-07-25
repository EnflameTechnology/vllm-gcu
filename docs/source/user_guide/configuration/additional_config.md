# Additional Configuration

`additional_config` is a flexible mechanism provided by vLLM that allows hardware plugins—such as `vllm-gcu`—to fine-tune backend behavior for optimized performance on GCU.

vLLM-GCU leverages this mechanism to enable features like graph mode execution, scheduler tuning, memory layout, and fused operators.

---

## How to Use

You can pass `additional_config` in both **online serving** and **offline inference** modes.

### Online Mode (via CLI)

```bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key": "config_value"}'
```

### Offline Mode (via Python)

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen3-8B", additional_config={"config_key": "config_value"})
```

---

## Available Configuration Options

| Config Name               | Type | Default | Description                                                               |
| ------------------------- | ---- | ------- | ------------------------------------------------------------------------- |
| `gcu_graph_config`        | dict | `{}`    | Graph mode config options for GCU backend                                 |
| `gcu_scheduler_config`    | dict | `{}`    | Scheduler-level tuning options for GCU                                    |
| `refresh`                 | bool | `False` | Force refresh of GCU plugin's global config (used in tests or RLHF loops) |
| `expert_map_path`         | str  | `None`  | Path to expert map when using MoE load balancing                          |
| `chunked_prefill_for_mla` | bool | `False` | Enables fused chunked prefill ops for MLA-enabled models                  |
| `kv_cache_dtype`          | str  | `None`  | Override kv-cache dtype. Set to `"int8"` to enable quantized cache        |

---

### `gcu_graph_config`

Used to enable and configure TorchAir-style graph optimizations for models that support GCU's static graph execution.

| Name                     | Type       | Default | Description                                                          |
| ------------------------ | ---------- | ------- | -------------------------------------------------------------------- |
| `enabled`                | bool       | `False` | Enables GCU static graph mode (supports DeepSeek, InternLM, etc.)    |
| `use_cached_graph`       | bool       | `False` | Whether to use cached compiled graph if available                    |
| `graph_batch_sizes`      | list\[int] | `[]`    | Explicit batch sizes to precompile for graph mode                    |
| `graph_batch_sizes_init` | bool       | `False` | Auto-detect batch sizes at runtime (if list is empty)                |
| `enable_multistream_moe` | bool       | `False` | Enables stream splitting for MoE models (shared expert optimization) |
| `enable_multistream_mla` | bool       | `False` | Enables offloading MLA vector ops to separate streams                |
| `enable_view_optimize`   | bool       | `True`  | Enables tensor view simplification in graph compiler                 |
| `enable_kv_nz`           | bool       | `False` | Use NZ-layout for kv-cache (GCU optimized layout)                    |

---

### `gcu_scheduler_config`

Overrides the default vLLM scheduler with GCU-aware scheduling logic. Also accepts fields from [vLLM’s `SchedulerConfig`](https://docs.vllm.ai/en/latest/api/vllm/config.html#vllm.config.SchedulerConfig).

| Name                     | Type | Default | Description                                   |
| ------------------------ | ---- | ------- | --------------------------------------------- |
| `enabled`                | bool | `False` | Enables custom GCU scheduler backend          |
| `enable_chunked_prefill` | bool | `False` | Enables chunked prefill pipeline optimization |

---

## Example Configuration

```json
{
  "gcu_graph_config": {
    "enabled": true,
    "use_cached_graph": true,
    "graph_batch_sizes": [1, 2, 4, 8],
    "graph_batch_sizes_init": false,
    "enable_multistream_moe": true,
    "enable_kv_nz": true
  },
  "gcu_scheduler_config": {
    "enabled": true,
    "enable_chunked_prefill": true
  },
  "refresh": false,
  "kv_cache_dtype": "int8"
}
```
