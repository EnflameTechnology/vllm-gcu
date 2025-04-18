def register():
    import contextlib

    from vllm import ModelRegistry

    from vllm_gcu.models.roberta.roberta import RobertaForSequenceClassification
    ModelRegistry.register_model("XLMRobertaForSequenceClassification", RobertaForSequenceClassification)
