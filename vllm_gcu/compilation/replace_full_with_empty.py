#!/usr/bin/env python
# coding=utf-8
import torch
from torch.fx import Node


def _is_full_node(node: Node):
    return (node.op == "call_function" and hasattr(node.target, "__name__")
            and "full" in str(node.target))


def _is_alias_node(node: Node):
    return (node.op == "call_function"
            and node.target == getattr(torch.ops.aten, "alias", None))


def _is_attention_node(node: Node):
    return (node.op == "call_function"
            and "auto_functionalized" in str(node.target)
            and "unified_attention_with_output" in node.args[0].name())


def _is_only_used_by_target(node: Node, target_node: Node) -> bool:
    visited = set()
    nodes_to_check = [node]

    while nodes_to_check:
        current_node = nodes_to_check.pop(0)
        if current_node in visited:
            continue
        visited.add(current_node)

        for user in current_node.users:
            if user == target_node:
                continue
            if _is_alias_node(user):
                nodes_to_check.append(user)
            elif user.op in ["output", "placeholder"]:
                continue
            else:
                return False

    return True


def _replace_full_with_empty(graph: torch.fx.Graph, full_node: Node):
    full_args = list(full_node.args)
    full_kwargs = dict(full_node.kwargs)

    empty_kwargs = {}
    for k, v in full_kwargs.items():
        if k not in ["fill_value", "value"]:
            empty_kwargs[k] = v

    with graph.inserting_before(full_node):
        empty_node = graph.call_function(
            torch.ops.aten.empty.memory_format,
            args=(full_args[0], ),
            kwargs=empty_kwargs,
        )

    full_node.replace_all_uses_with(empty_node)

    graph.erase_node(full_node)


def replace_full_with_empty(graph: torch.fx.Graph):
    unified_attention_nodes = []
    for node in graph.nodes:
        if _is_attention_node(node):
            unified_attention_nodes.append(node)

    if not unified_attention_nodes:
        return

    full_nodes_to_replace = set()

    for attention_node in unified_attention_nodes:
        for arg in attention_node.args:
            if isinstance(
                    arg,
                    Node) and _is_full_node(arg) and _is_only_used_by_target(
                        arg, attention_node):
                full_nodes_to_replace.add(arg)

        for key, arg in attention_node.kwargs.items():
            if key == "output" and isinstance(
                    arg,
                    Node) and _is_full_node(arg) and _is_only_used_by_target(
                        arg, attention_node):
                full_nodes_to_replace.add(arg)

    for full_node in full_nodes_to_replace:
        _replace_full_with_empty(graph, full_node)
