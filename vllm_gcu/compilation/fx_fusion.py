from typing import Any, List, Callable, Optional
from collections import defaultdict

import torch


def get_arg(node, index):
    if len(node.args) > 0:
        arg = node.args[index]
    else:
        # TODO
        assert False
        arg = node.kwargs["input"]
    return arg


class Fusion:
    def __init__(self, meta) -> None:
        self.op1, self.op2, self.fuse_op = meta[0:3]
        self.out_in_map = meta[3]
        self.fuse_op_args = meta[4]

    def __call__(self, module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        class Match:
            op1: Optional[torch.fx.Node] = None
            op2: Optional[torch.fx.Node] = None

        matches = defaultdict(Match)
        for node in module.graph.find_nodes(op="call_function", target=self.op2):
            args = []
            for i in self.out_in_map:
                args.append(get_arg(node, i[1]))
            arg_hash = hash(tuple(args))
            matches[arg_hash].op2 = node
        for node in module.graph.find_nodes(op="call_function", target=self.op1):
            args = []
            for i in self.out_in_map:
                args.append(get_arg(node, i[0]))
            arg_hash = hash(tuple(args))
            assert matches[arg_hash].op1 is None, "duplicated match"
            matches[arg_hash].op1 = node
        for i in matches:
            if matches[i].op1 is None or matches[i].op2 is None:
                continue

            with module.graph.inserting_before(matches[i].op2):
                args = []
                for op_idx, op_arg_idx in self.fuse_op_args:
                    if op_idx == 0:
                        args.append(matches[i].op1.args[op_arg_idx])
                    else:
                        args.append(matches[i].op2.args[op_arg_idx])
                fused_node = module.graph.call_function(
                    self.fuse_op, args=tuple(args)
                )
                matches[i].op2.replace_all_uses_with(fused_node)
                module.graph.erase_node(matches[i].op2)
                module.graph.erase_node(matches[i].op1)

        module.graph.lint()
        module.recompile()
        return module


class MergeQuant:
    '''Merge dynamic quant ops with same inputs'''

    def __init__(self, meta) -> None:
        self.op, self.inputs = meta

    def __call__(self, module: torch.fx.GraphModule) -> torch.fx.GraphModule:
        matches = defaultdict(list)
        for node in module.graph.find_nodes(op="call_function", target=self.op):
            args = []
            for i in self.inputs:
                args.append(get_arg(node, i))
            arg_hash = hash(tuple(args))
            matches[arg_hash].append(node)

        for i in matches:
            quant_ops = matches[i]
            if len(quant_ops) <= 1:
                continue

            for node in quant_ops[1:]:
                for idx, arg in enumerate(node.args):
                    if arg != quant_ops[0].args[idx]:
                        assert isinstance(arg, torch.fx.Node)
                        arg.replace_all_uses_with(quant_ops[0].args[idx])
                module.graph.erase_node(node)
        module.graph.lint()
        module.recompile()
        return module


def get_passes():
    merge_quant_patterns = [
        (torch.ops._C.dynamic_per_token_group_fp8_quant, [2, 3])
    ]

    fusion_patterns = [
        (
            torch.ops._C.silu_and_mul,
            torch.ops._C.dynamic_per_token_group_fp8_quant,
            torch.ops._C.silu_mul_per_token_group_quant,
            [(0, 2)],
            [(1, 0), (1, 1), (0, 1), (1, 3)],
        ),
        (
            torch.ops._C.fused_add_rms_norm,
            torch.ops._C.dynamic_per_token_group_fp8_quant,
            torch.ops._C.fused_add_rms_norm_per_token_group_quant_fp8,
            [(0, 2)],
            [(1, 0), (0, 1), (1, 1), (0, 0), (0, 2), (0, 3), (1, 3)]
        ),
        (
            torch.ops._C.rms_norm,
            torch.ops._C.dynamic_per_token_group_fp8_quant,
            torch.ops._C.rms_norm_per_token_group_quant_fp8,
            [(0, 2)],
            [(1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3)]

        ),
        (
            torch.ops._C.silu_and_mul_pad,
            torch.ops._C.dynamic_per_token_group_fp8_quant,
            torch.ops._C.silu_mul_per_token_group_quant_with_size,
            [(0, 2)],
            [(1, 0), (1, 1), (0, 1), (0, 2), (1, 3)]
        ),
    ]

    passes = []
    passes += [MergeQuant(i) for i in merge_quant_patterns]
    passes += [Fusion(i) for i in fusion_patterns]
    return passes
