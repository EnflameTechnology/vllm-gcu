#!/usr/bin/env python
# coding=utf-8
import vllm_gcu.patch.patch_0_10_2.entrypoints  # noqa
import vllm_gcu.patch.patch_0_10_2.moe_layer  # noqa
import vllm_gcu.patch.patch_0_10_2.flash_attn  # noqa
import vllm_gcu.patch.patch_0_10_2.compilation_backends  # noqa
import vllm_gcu.patch.patch_0_10_2.utils  # noqa
import vllm_gcu.patch.patch_0_10_2.forward_context  # noqa
import vllm_gcu.patch.patch_0_10_2.noop_elimination  # noqa
import vllm_gcu.patch.patch_0_10_2.rejection_sampler
import vllm_gcu.patch.patch_0_10_2.flashmla  # noqa
import vllm_gcu.patch.patch_0_10_2.compressed_tensor  # noqa
import vllm_gcu.patch.patch_0_10_2.modular_kernel  # noqa
