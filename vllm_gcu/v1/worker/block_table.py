# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.logger import init_logger
from vllm.utils import cdiv

class DoubleBlockTable(BlockTable):
    """
        拥有两份BlockTable缓存区的BlockTable，并且增加了一个切换缓存区的函数
    """
    def __init__(
        self,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device    
        
        self.buffer_id = 0
        
        self.all_block_table = torch.zeros(
            (2, max_num_reqs, max_num_blocks_per_req),
            device=self.device,
            dtype=torch.int32,
        )
        self.block_table = self.all_block_table[0]

        self.block_table_cpu = torch.zeros(
            (max_num_reqs, max_num_blocks_per_req),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        
        self.block_table_np = self.block_table_cpu.numpy()

        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        self.slot_mapping_cpu = torch.zeros(self.max_num_batched_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

        self.all_slot_mapping = torch.zeros((2, self.max_num_batched_tokens),
                                        dtype=torch.int64,
                                        device=self.device)
        self.slot_mapping = self.all_slot_mapping[0]
        
    def switch_buffer(self, buffer_id=None):
        if self.buffer_id == buffer_id:
            return
        if buffer_id is None:
            self.buffer_id = (self.buffer_id + 1) % 2
        else:
            self.buffer_id = buffer_id
        self.block_table = self.all_block_table[self.buffer_id]
        self.slot_mapping = self.all_slot_mapping[self.buffer_id]



class DoubleMultiGroupBlockTable(MultiGroupBlockTable):
    """The BlockTables for each KV cache group."""

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: torch.device, block_sizes: list[int]) -> None:
        self.block_tables = [
            DoubleBlockTable(max_num_reqs, cdiv(max_model_len, block_size),
                       max_num_batched_tokens, pin_memory, device)
            for block_size in block_sizes
        ]

    def switch_buffer(self, buffer_id=None) -> None:
        for block_table in self.block_tables:
            block_table.switch_buffer(buffer_id)

