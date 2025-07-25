# Adding a New Model

This guide explains how to integrate a new or customized model into **vLLM-GCU**. It aligns with the [official vLLM guide on model integration](https://docs.vllm.ai/en/stable/contributing/model/), and includes GCU-specific implementation details.

---

## Step 1: Implementing Your Model

You can either **build from scratch** or **extend existing vLLM models** to support your custom model.

Before you begin:

* Check whether your model is already implemented in [vLLM's model directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models).
* Prefer adapting existing models if they share a similar architecture.

---

### Option A: Implement a New Model from Scratch

Follow the structure of the [vLLM OPT model guide](https://docs.vllm.ai/en/stable/contributing/model/basic.html) as a reference.

#### 1. File Structure

Place your model implementation in the `vllm_gcu/models/` directory.

Required modules for a **decoder-only LLM**:

| Module Class         | Responsibility                     |
| -------------------- | ---------------------------------- |
| `*ModelForCausalLM`  | Top-level wrapper for inference    |
| `*Model`             | Main model architecture            |
| `*DecoderLayer`      | Transformer block                  |
| `*Attention`, `*MLP` | Component-level computation layers |

> **Note**: `*` denotes your model’s prefix (e.g., `Custom`, `MyModel`).

#### 2. Required Interfaces

Each module must be initialized with a `prefix` argument to ensure compatibility with vLLM’s weight loading and quantization system.

Required methods:

| Module              | Required Methods                                         |
| ------------------- | -------------------------------------------------------- |
| `*ModelForCausalLM` | `get_input_embeddings`, `compute_logits`, `load_weights` |
| `*Model`            | `get_input_embeddings`, `forward`, `load_weights`        |

#### 3. Attention Integration

Import and use vLLM’s backend-aware attention interface:

```python
from vllm.attention import Attention
```

This enables backend-specific routing (e.g., GCU via `get_attn_backend_cls()` in `vllm_gcu/platform.py`).

#### 4. Tensor Parallelism

Use vLLM’s built-in parallel layers for distributed execution:

* `ColumnParallelLinear`
* `RowParallelLinear`
* `VocabParallelEmbedding`
* `ParallelLMHead`

GCU-optimized layers (like `RMSNorm`) are found in `vllm_gcu/ops/`.

#### 5. Example Implementation

```python
class CustomAttention(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.attn = Attention(prefix=f"{prefix}.attn")

class CustomDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.self_attn = CustomAttention(vllm_config, prefix=f"{prefix}.self_attn")

class CustomModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(vllm_config, prefix=f"{prefix}.layers.{i}")
            for i in range(vllm_config.model_config.hf_config.num_hidden_layers)
        ])

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        ...

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ...

class CustomModelForCausalLM(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.model = CustomModel(vllm_config, prefix=f"{prefix}.model")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        ...

    def forward(...): ...
    def compute_logits(...): ...
    def load_weights(...): ...
```

---

### Option B: Extend an Existing Model

If your model shares architecture with a supported vLLM model, inherit from it and override key logic.

#### Example: Customize DeepseekV2

```python
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM

class CustomDeepseekV2ForCausalLM(DeepseekV2ForCausalLM):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts": ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def forward(...): ...
```

---

## Step 2: Registering Your Model

Use the vLLM plugin system to register your model externally.

### 1. Edit `vllm_gcu/models/__init__.py`:

```python
from vllm import ModelRegistry

def register_model():
    from .custom_model import CustomModelForCausalLM
    from .deepseek_v2 import CustomDeepseekV2ForCausalLM

    # Register new architecture
    ModelRegistry.register_model(
        "CustomModelForCausalLM",
        "vllm_gcu.models.custom_model:CustomModelForCausalLM"
    )

    # Override existing one
    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_gcu.models.deepseek_v2:CustomDeepseekV2ForCausalLM"
    )
```

### 2. Match `architectures` in `config.json`

```json
{
  "architectures": [
    "CustomModelForCausalLM"
  ]
}
```

---

## Step 3: Verification

### Case A: Overriding Existing Architecture

You will see a log like this:

```bash
Model architecture DeepseekV2ForCausalLM is already registered...
```

### Case B: Registering a New Architecture

To confirm registration, add a log line:

```python
logger.info(f"model_arch: {model_arch} has been registered here!")
```

---

## Step 4: Testing

Thoroughly test the new model for:

* **Basic functionality**
* **Offline/online inference**
* **Accuracy**
* **Performance benchmarks**

Refer to:

* [Accuracy testing guide](https://vllm-gcu.readthedocs.io/en/latest/developer_guide/evaluation/index.html)
* [Performance benchmark guide](https://vllm-gcu.readthedocs.io/en/latest/developer_guide/performance/benchmark.html)

---

## Step 5: Document Support

Add your model to the [Supported Models List](https://vllm-gcu.readthedocs.io/en/latest/user_guide/support_matrix/supported_models.html) once it is validated.
