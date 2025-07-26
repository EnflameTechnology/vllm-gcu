#!/usr/bin/env python
# coding=utf-8

import copy
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import torch
import torch.fx as fx
import torch_gcu
import vllm.envs as envs
from torch._inductor.codegen.common import device_codegens, get_scheduling_for_device
from torch._inductor.codegen.triton import TritonScheduling
from vllm.compilation.compiler_interface import (
    AlwaysHitShapeEnv,
    InductorAdaptor,
)
from vllm.compilation.inductor_pass import pass_context

from vllm_gcu.utils import is_torch_equal_or_newer

try:
    if device_schedule := get_scheduling_for_device("gcu") == TritonScheduling:
        device_codegens.pop("gcu")
    import tops_extension.torch  # noqa
except Exception:
    pass


@contextmanager
def version():
    is_cuda_version = torch.version.cuda
    if not is_cuda_version:
        torch.version.cuda = torch_gcu.version.gcu
    yield
    if not is_cuda_version:
        torch.version.cuda = None


@contextmanager
def lowering():
    import torch._inductor.lowering as til

    BLACK_LIST = ["name"]
    origin_lowerings = {}
    skip_lowerings = []

    for name in dir(torch.ops.aten):
        if not name.startswith("_") and name not in BLACK_LIST:
            op = getattr(torch.ops.aten, name)

            if isinstance(op, torch._ops.OpOverloadPacket):
                for ol in op.overloads():
                    op_overload = til.get_overloads(getattr(op, ol))
                    if op_overload[0] in til.lowerings:
                        origin_lowerings.update(
                            dict.fromkeys(op_overload, til.lowerings[op_overload[0]])
                        )
                    else:
                        skip_lowerings.append(op_overload[0])
            elif isinstance(
                op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
            ):
                op_overload = til.get_overloads(op)
                if op_overload[0] in til.lowerings:
                    origin_lowerings.update(
                        dict.fromkeys(op_overload, til.lowerings[op_overload[0]])
                    )
                else:
                    skip_lowerings.append(op_overload[0])

            til.make_fallback(op, warn=False, override_decomp=True)

    for name in dir(torch.ops.prim):
        if not name.startswith("_") and name not in BLACK_LIST:
            op = getattr(torch.ops.prim, name)
            til.make_fallback(op, warn=False, override_decomp=True)

    yield

    for op_overload, _ in origin_lowerings.items():
        til.register_lowering(op_overload, type_promotion_kind=None)(
            origin_lowerings[op_overload]
        )
    for op_overload in skip_lowerings:
        til.lowerings.pop(op_overload)


class CustomInductorAdaptor(InductorAdaptor):
    def compute_hash(self, vllm_config) -> str:
        with version():
            return super().compute_hash(vllm_config)

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: List[Any],
        compiler_config: Dict[str, Any],
        runtime_shape: Optional[int] = None,
        key: Optional[str] = None,
    ) -> Tuple[Optional[Callable], Optional[Any]]:
        from torch._inductor import config

        current_config = config.get_config_copy()
        from torch._inductor.compile_fx import compile_fx

        # for _pass in get_passes():
        #     graph = _pass(graph)

        # disable remote cache
        # current_config["force_disable_caches"] = True
        current_config["fx_graph_cache"] = True
        current_config["fx_graph_remote_cache"] = False

        if compiler_config is not None:
            current_config.update(compiler_config)

        if isinstance(runtime_shape, int):
            # for a specific batchsize, tuning triton kernel parameters
            # can be beneficial
            current_config["max_autotune"] = True
            current_config["coordinate_descent_tuning"] = True

        # inductor can inplace modify the graph, so we need to copy it
        # see https://github.com/pytorch/pytorch/issues/138980
        graph = copy.deepcopy(graph)

        # it's the first time we compile this graph
        # the assumption is that we don't have nested Inductor compilation.
        # compiled_fx_graph_hash will only be called once, and we can hook
        # it to get the hash of the compiled graph directly.

        hash_str, file_path = None, None
        from torch._inductor.codecache import compiled_fx_graph_hash, FxGraphCache

        if torch.__version__.startswith("2.5"):
            original_load = FxGraphCache.load
            original_load_name = "torch._inductor.codecache.FxGraphCache.load"

            def hijack_load(*args, **kwargs):
                inductor_compiled_graph = original_load(*args, **kwargs)
                nonlocal file_path
                compiled_fn = inductor_compiled_graph.current_callable
                file_path = compiled_fn.__code__.co_filename  # noqa
                if not file_path.startswith(self.cache_dir):
                    # hooked in the align_inputs_from_check_idxs function
                    # in torch/_inductor/utils.py
                    for cell in compiled_fn.__closure__:
                        if not callable(cell.cell_contents):
                            continue
                        if cell.cell_contents.__code__.co_filename.startswith(
                            self.cache_dir
                        ):
                            # this is the real file path compiled from Inductor
                            file_path = cell.cell_contents.__code__.co_filename
                            break
                return inductor_compiled_graph

            def hijacked_compile_fx_inner(*args, **kwargs):
                with lowering():
                    output = torch._inductor.compile_fx.compile_fx_inner(
                        *args, **kwargs
                    )
                return output

        elif is_torch_equal_or_newer("2.6"):
            # function renamed in 2.6
            original_load_name = None
            import vllm_gcu.patch.torch_2_6_0.refs_pad  # noqa: F401

            def hijacked_compile_fx_inner(*args, **kwargs):
                with version(), lowering():
                    output = torch._inductor.compile_fx.compile_fx_inner(
                        *args, **kwargs
                    )
                    nonlocal hash_str
                    inductor_compiled_graph = output
                    if inductor_compiled_graph is not None:
                        nonlocal file_path
                        file_path = (
                            inductor_compiled_graph.current_callable.__code__.co_filename
                        )  # noqa
                        compiled_fn = inductor_compiled_graph.current_callable
                        file_path = compiled_fn.__code__.co_filename  # noqa
                        if not file_path.startswith(self.cache_dir):
                            # hooked in the align_inputs_from_check_idxs function
                            # in torch/_inductor/utils.py
                            if compiled_fn.__closure__:
                                for cell in compiled_fn.__closure__:
                                    if not callable(cell.cell_contents):
                                        continue
                                    code = cell.cell_contents.__code__
                                    if code.co_filename.startswith(self.cache_dir):
                                        # this is the real file path
                                        # compiled from Inductor
                                        file_path = code.co_filename
                                        break
                            hash_str = inductor_compiled_graph._fx_graph_cache_key
                    return output

        def hijack_compiled_fx_graph_hash(*args, **kwargs):
            with version():
                out = compiled_fx_graph_hash(*args, **kwargs)
            nonlocal hash_str
            hash_str = out[0]
            return out

        def _check_can_cache(*args, **kwargs):
            # no error means it can be cached.
            # Inductor refuses to cache the graph outside of Dynamo
            # tracing context, and also disables caching for graphs
            # with high-order ops.
            # For vLLM, in either case, we want to cache the graph.
            # see https://github.com/pytorch/pytorch/blob/9f5ebf3fc609105a74eab4ccc24932d6353ff566/torch/_inductor/codecache.py#L1221 # noqa
            return

        def _get_shape_env() -> AlwaysHitShapeEnv:
            return AlwaysHitShapeEnv()

        def _should_reinplace_scatter(node):
            return False

        with ExitStack() as stack:
            # hijack to get the compiled graph itself
            if original_load_name is not None:
                stack.enter_context(patch(original_load_name, hijack_load))

            # for hijacking the hash of the compiled graph
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.compiled_fx_graph_hash",
                    hijack_compiled_fx_graph_hash,
                )
            )

            # for providing a dummy shape environment
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._get_shape_env",
                    _get_shape_env,
                )
            )

            # for forcing the graph to be cached
            stack.enter_context(
                patch(
                    "torch._inductor.codecache.FxGraphCache._check_can_cache",
                    _check_can_cache,
                )
            )

            stack.enter_context(
                patch(
                    "torch._inductor.fx_passes.reinplace.should_reinplace_scatter",
                    _should_reinplace_scatter
                )
            )

            stack.enter_context(self.metrics_context())

            if is_torch_equal_or_newer("2.6"):
                stack.enter_context(
                    torch._inductor.config.patch(fx_graph_remote_cache=False)
                )
                stack.enter_context(
                    torch._functorch.config.patch(enable_remote_autograd_cache=False)
                )

            with pass_context(runtime_shape):
                compiled_graph = compile_fx(
                    graph,
                    example_inputs,
                    inner_compile=hijacked_compile_fx_inner,
                    config_patches=current_config,
                )

        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            assert hash_str is not None, "failed to get the hash of the compiled graph"
            assert (
                file_path is not None
            ), "failed to get the file path of the compiled graph"
        return compiled_graph, (hash_str, file_path)
