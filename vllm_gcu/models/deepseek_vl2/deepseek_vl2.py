from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.deepseek_vl2 import (DeepseekVLV2ForCausalLM,
    DeepseekVL2MultiModalProcessor, DeepseekVL2ProcessingInfo,
    DeepseekVL2DummyInputsBuilder)


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekVL2MultiModalProcessor,
    info=DeepseekVL2ProcessingInfo,
    dummy_inputs=DeepseekVL2DummyInputsBuilder)
class DeepseekVLV2ForCausalLMGCU(DeepseekVLV2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.tile_tag == "2D":
            # invalid spellchecker fix
            # https://github.com/vllm-project/vllm/pull/20618
            self.view_seperator = self.view_separator
            del self.view_separator
            self.view_separator = self.view_seperator.data
