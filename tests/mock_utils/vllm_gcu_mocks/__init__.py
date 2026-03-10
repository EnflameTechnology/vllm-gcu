
def register_mock_models():
    from vllm import ModelRegistry
    ModelRegistry.register_model("DeepSeekMTPModel", "vllm_gcu_mocks.mock_positions_mtp:MockPosMTP")
    ModelRegistry.register_model("MockPosForCausalLM", "vllm_gcu_mocks.mock_positions:MockPosForCausalLM")