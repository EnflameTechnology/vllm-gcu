from unittest.mock import patch
from vllm_gcu.attention.backends.indexer import get_builder_cls

patch("vllm.v1.attention.backends.mla.indexer.DeepseekV32IndexerBackend.get_builder_cls", \
    get_builder_cls).start()