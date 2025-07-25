# Multi-Modal Support

This guide walks you through extending a basic model to support multi-modal inputs in vLLM.

---

## 1. Extend the Base vLLM Model

Start by implementing your model in vLLM following [these steps](new_model.md). Then, update the model to support multi-modal inputs:

### Implement `get_placeholder_str`

Define the placeholder string representing a multi-modal item in the text prompt. This must align with your model's chat template.

```python
class YourModelForImage2Seq(nn.Module):
    ...

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")
```

### Update `forward` Method

Add a keyword argument for each multi-modal input (e.g., image tensors):

```diff
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
+   pixel_values: torch.Tensor,
) -> SamplerOutput:
```

Alternatively, use `**kwargs` and extract the needed inputs dynamically.

### Implement `get_multimodal_embeddings`

Return embeddings for multi-modal inputs using your tokenizer and encoder:

```python
class YourModelForImage2Seq(nn.Module):
    ...

    def _process_image_input(self, image_input: YourModelImageInputs) -> torch.Tensor:
        assert self.vision_encoder is not None
        image_features = self.vision_encoder(image_input)
        return self.multi_modal_projector(image_features)

    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)
```

!!! important
Return value must be a 3D tensor `(num_items, feature_size, hidden_size)` or a list of 2D tensors `(feature_size, hidden_size)`.

### Implement `get_input_embeddings`

Merge text and multi-modal embeddings:

```python
from .utils import merge_multimodal_embeddings

class YourModelForImage2Seq(nn.Module):
    ...

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                placeholder_token_id=self.config.image_token_index
            )

        return inputs_embeds
```

### Implement `get_language_model`

Provide access to the underlying language model:

```python
def get_language_model(self) -> torch.nn.Module:
    return self.language_model
```

### Register `SupportsMultiModal`

Finally, inherit from `SupportsMultiModal`:

```diff
+ from vllm.model_executor.models.interfaces import SupportsMultiModal

- class YourModelForImage2Seq(nn.Module):
+ class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

---

## 2. Define Processing Information

Create a subclass of `BaseProcessingInfo` to specify modality limits:

```python
def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
    return {"image": None, "video": 1}
```

---

## 3. Provide Dummy Inputs

Inherit from `BaseDummyInputsBuilder` to generate worst-case dummy inputs for memory profiling.

### Define `get_dummy_text` and `get_dummy_mm_data`

Ensure the generated inputs stress the model's memory limits (e.g., maximize image token count).

Example for LLaVA and Fuyu are already included in detail in your original document.

---

## 4. Configure the Processor

Subclass `BaseMultiModalProcessor` to connect HF processing logic.

### Define `_get_mm_fields_config`

Return the structure of tensor fields corresponding to each modality.

Example for LLaVA:

```python
def _get_mm_fields_config(
    self,
    hf_inputs: BatchFeature,
    hf_processor_mm_kwargs: Mapping[str, object],
) -> Mapping[str, MultiModalFieldConfig]:
    return dict(
        pixel_values=MultiModalFieldConfig.batched("image"),
    )
```

### Define `_get_prompt_updates`

Describe how the HF processor modifies the prompt. Return a list of `PromptUpdate` entries.

Examples for LLaVA and Fuyu are detailed in the original doc and should be adapted as needed per model.

---

## 5. Register the Processor

Link your processor-related classes using the registry:

```diff
from vllm.model_executor.models.interfaces import SupportsMultiModal
+ from vllm.multimodal import MULTIMODAL_REGISTRY

+ @MULTIMODAL_REGISTRY.register_processor(
+     YourMultiModalProcessor,
+     info=YourProcessingInfo,
+     dummy_inputs=YourDummyInputsBuilder
+ )
class YourModelForImage2Seq(nn.Module, SupportsMultiModal):
```

---

## Notes

### Inserting Tokens Instead of Replacing

Use `PromptInsertion` when inserting tokens (e.g., BLIP-2, Florence2, Molmo).

### Other Prompt Updates

If the HF processor applies changes regardless of multi-modal inputs (e.g., adding separators), override `_apply_hf_processor_tokens_only`.

Examples:

* Fuyu appends `boa_token`
* Chameleon appends `sep_token`
* Molmo defines a custom chat template

### Custom HF Processor

If your model lacks a HuggingFace processor, create a compatible one and use it with `_call_hf_processor`.