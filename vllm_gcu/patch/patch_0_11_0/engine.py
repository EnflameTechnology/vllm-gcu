from unittest.mock import patch


def post_step(self, model_executed: bool) -> None:
    use_async_scheduling = self.vllm_config.scheduler_config.async_scheduling
    if not use_async_scheduling and self.use_spec_decode and model_executed:
        # Take the draft token ids.
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)


patch("vllm.v1.engine.core.EngineCore.post_step", post_step).start()
