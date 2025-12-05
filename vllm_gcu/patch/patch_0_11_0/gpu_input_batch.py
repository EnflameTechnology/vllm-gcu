from unittest.mock import patch
from vllm_gcu.worker.gcu_input_batch import GCUInputBatch


patch("vllm.v1.worker.gpu_input_batch.InputBatch", GCUInputBatch).start()
patch("vllm.v1.worker.gpu_model_runner.InputBatch", GCUInputBatch).start()
