# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import get_world_group

from vllm.logger import init_logger


from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import P2pNcclConnector
from vllm_gcu.distributed.kv_transfer.v1.p2p_eccl_engine import P2pEcclEngine


logger = init_logger(__name__)


class P2pEcclConnector(P2pNcclConnector):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super(P2pNcclConnector, self).__init__(vllm_config=vllm_config, role=role)
        self._block_size = vllm_config.cache_config.block_size
        self._requests_need_load: dict[str, Any] = {}
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer
        self.chunked_prefill: dict[str, Any] = {}

        self._rank = get_world_group().rank \
            if role == KVConnectorRole.WORKER else 0
        self._local_rank = get_world_group().local_rank \
            if role == KVConnectorRole.WORKER else 0

        self.p2p_nccl_engine = P2pEcclEngine(
            local_rank=self._local_rank,
            config=self.config,
            hostname="",
            port_offset=self._rank,
        ) if role == KVConnectorRole.WORKER else None
