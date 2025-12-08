from unittest.mock import patch
import numpy as np
import torch
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
import vllm_gcu.envs as gcu_envs

def compute_slot_mapping_device(self, req_indices: np.ndarray,
                             positions: torch.Tensor) -> None:
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
    # where K is the max_num_blocks_per_req and the block size is 2.
    # NOTE(woosuk): We can't simply use `token_indices // block_size`
    # here because M (max_model_len) is not necessarily divisible by
    # block_size.
    if self.dcp_world_size > 1:
        # Note(hc): The DCP implement store kvcache with an interleave
        # style, the kvcache for the token whose token_idx is i is
        # always stored on the GPU whose dcp_rank equals i % cp_world_size:

        # Use a "virtual block" which equals to world_size * block_size
        # for block_table_indices calculation.
        virtual_block_size = self.block_size * self.dcp_world_size
        #block_table_indices = (req_indices * self.max_num_blocks_per_req +
        #                       positions // virtual_block_size)
        block_table_indices = torch.tensor(req_indices.tolist(),
                                            dtype = torch.int64,
                                            pin_memory = self.pin_memory).to(
                                            self.device,
                                            non_blocking = True
                                            )
        block_table_indices.mul_(self.max_num_blocks_per_req)
        block_table_indices.add_(positions.div(virtual_block_size, 
                                                rounding_mode = "floor"))
        #block_numbers = self.block_table_np.ravel()[block_table_indices]
        block_numbers = torch.index_select(self.block_table.gpu.flatten(), 
                                            dim = 0,
                                            index = block_table_indices)
        # Use virtual_block_size for mask calculation, which marks local
        # tokens.
        #virtual_block_offsets = positions % virtual_block_size
        virtual_block_offsets = torch.remainder(positions, self.block_size)
        virtual_block_rank = torch.remainder(virtual_block_offsets,
                                                self.dcp_world_size)
        mask = virtual_block_rank == self.dcp_rank
        # Calculate local block_offsets
        block_offsets = virtual_block_offsets.div(self.dcp_world_size,
                                                    rounding_mode="floor")
        # Calculate slot_mapping
        slot_mapping = block_numbers * self.block_size + block_offsets
        # Write final slots, use -1 for not-local
        self.slot_mapping.gpu[:req_indices.shape[0]] = torch.where(
            mask, slot_mapping, -1)
    else:
        #block_table_indices = (req_indices * self.max_num_blocks_per_req +
        #                       positions // self.block_size)
        block_table_indices = torch.tensor(req_indices.tolist(),
                                            dtype = torch.int64,
                                            pin_memory = self.pin_memory).to(
                                            self.device,
                                            non_blocking = True
                                            )
        block_table_indices.mul_(self.max_num_blocks_per_req)
        block_table_indices.add_(positions.div(self.block_size, 
                                                rounding_mode = "floor"))
        #block_numbers = self.block_table_np.ravel()[block_table_indices]
        block_numbers = torch.index_select(self.block_table.gpu.flatten(), 
                                            dim = 0,
                                            index = block_table_indices)
        #block_offsets = positions % self.block_size
        block_offsets = torch.remainder(positions, self.block_size)
        #np.add(block_numbers * self.block_size,
        #       block_offsets,
        #       out=self.slot_mapping_np[:req_indices.shape[0]])
        block_numbers.mul_(self.block_size)
        torch.add(block_numbers, block_offsets, 
                    out = self.slot_mapping.gpu[:req_indices.shape[0]])
        
def multi_compute_slot_mapping_device(self, req_indices: np.ndarray,
                             positions: torch.Tensor) -> None:
    for block_table in self.block_tables:
        block_table.compute_slot_mapping_device(req_indices, positions)

if gcu_envs.VLLM_GCU_ENABLE_DEEPSEEK_MTP_FUSION:
    patch.object(BlockTable, "compute_slot_mapping_device", compute_slot_mapping_device, create=True).start()
    patch.object(MultiGroupBlockTable, "compute_slot_mapping_device", multi_compute_slot_mapping_device, create=True).start()


def _compute_slot_mapping(
    self,
    req_indices: torch.Tensor,
    positions: torch.Tensor
) -> None:
    if isinstance(req_indices, torch.Tensor) and isinstance(positions, torch.Tensor):
        if self.dcp_world_size > 1:
            virtual_block_size = self.block_size * self.dcp_world_size
            block_table_indices = (req_indices * self.max_num_blocks_per_req + positions // virtual_block_size)
            block_numbers = self.block_table.gpu.flatten()[block_table_indices]
            virtual_block_offsets = positions % virtual_block_size
            mask = virtual_block_offsets % self.dcp_world_size == self.dcp_rank
            block_offsets = virtual_block_offsets // self.dcp_world_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.gpu[:req_indices.shape[0]] = torch.where(mask, slot_mapping, -1)
        else:
            block_table_indices = (req_indices * self.max_num_blocks_per_req + positions // self.block_size)
            block_numbers = self.block_table.gpu.flatten()[block_table_indices]
            block_offsets = positions % self.block_size
            torch.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.gpu[:req_indices.shape[0]])
    else:
        if self.dcp_world_size > 1:
            virtual_block_size = self.block_size * self.dcp_world_size
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // virtual_block_size)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            virtual_block_offsets = positions % virtual_block_size
            mask = virtual_block_offsets % self.dcp_world_size == self.dcp_rank
            block_offsets = virtual_block_offsets // self.dcp_world_size
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.np[:req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1)
        else:
            block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                   positions // self.block_size)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size,
                   block_offsets,
                   out=self.slot_mapping.np[:req_indices.shape[0]])


patch("vllm.v1.worker.block_table.BlockTable.compute_slot_mapping", _compute_slot_mapping).start()
