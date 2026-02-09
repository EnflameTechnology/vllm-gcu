#!/usr/bin/env python
# coding=utf-8
import torch
from dataclasses import dataclass
from unittest.mock import patch


@dataclass
class GCUMLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None

patch("vllm.model_executor.layers.mla.MLAModules", GCUMLAModules).start()
