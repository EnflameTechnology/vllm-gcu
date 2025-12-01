# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.fx as fx
import vllm.envs as envs
import vllm_gcu.envs as gcu_envs
from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import patch
from vllm.compilation.backends import VllmBackend
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import end_monitoring_torch_compile
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils import weak_ref_tensors
from vllm.compilation.cuda_piecewise_backend import ConcreteSizeEntry, CUDAPiecewiseBackend
logger = init_logger(__name__)


class DoubleCUDAPiecewiseBackend(CUDAPiecewiseBackend):

    def __init__(self, graph: fx.GraphModule, vllm_config: VllmConfig,
                 graph_pool: Any, piecewise_compile_index: int,
                 total_piecewise_compiles: int, sym_shape_indices: list[int],
                 compiled_graph_for_general_shape: Callable,
                 vllm_backend: VllmBackend):
        """
        The backend for piecewise compilation.
        It mainly handles the compilation and cudagraph capturing.

        We will compile `self.graph` once for the general shape,
        and then compile for different shapes specified in
        `compilation_config.compile_sizes`.

        Independently, we will capture cudagraph for different shapes.

        If a shape needs both compilation and cudagraph, we will
        compile it first, and then capture cudagraph.
        """
        self.graph = graph
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.graph_pool = graph_pool
        self.piecewise_compile_index = piecewise_compile_index
        self.total_piecewise_compiles = total_piecewise_compiles
        self.vllm_backend = vllm_backend

        self.is_first_graph = piecewise_compile_index == 0
        self.is_last_graph = (
            piecewise_compile_index == total_piecewise_compiles - 1)

        self.compile_sizes: set[int] = set(
            self.compilation_config.compile_sizes)
        self.cudagraph_capture_sizes: set[int] = set(
            self.compilation_config.cudagraph_capture_sizes
        ) if self.compilation_config.use_cudagraph else set()

        self.first_run_finished = False

        self.compiled_graph_for_general_shape = compiled_graph_for_general_shape  # noqa

        self.sym_shape_indices = sym_shape_indices

        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # concrete_size_entries中的key变成 (int，int)，其中第一个int是
        # exec_buffer，第二个int是runtime shape
        self.concrete_size_entries: dict[(int,int), ConcreteSizeEntry] = {}

        async_executing = vllm_config.additional_config.get("async_executing", False)
        
        compile_size_capture_times = 1
        # 如果开启async_executing，每个compile_size需要捕获两次
        if async_executing:
            compile_size_capture_times = 2
        
        self.to_be_compiled_sizes: dict[int, int] = {
            compile_size:compile_size_capture_times for compile_size in self.compile_sizes
        }

        for shape in self.compile_sizes.union(self.cudagraph_capture_sizes):
            
            self.concrete_size_entries[(0, shape)] = ConcreteSizeEntry(
                runtime_shape=shape,
                need_to_compile=shape in self.compile_sizes,
                use_cudagraph=shape in self.cudagraph_capture_sizes,
            )
            
            # 只有开启异步执行时，才会创建第二个buffer
            if async_executing:
                self.concrete_size_entries[(1, shape)] = ConcreteSizeEntry(
                    runtime_shape=shape,
                    need_to_compile=shape in self.compile_sizes,
                    use_cudagraph=shape in self.cudagraph_capture_sizes,
                )

    def __call__(self, *args) -> Any:

        if not self.first_run_finished:
            self.first_run_finished = True
            self.check_for_ending_compilation()
            return self.compiled_graph_for_general_shape(*args)

        runtime_shape = args[self.sym_shape_indices[0]]

        # 从forawrd_context中获取 exec_buffer，
        # 如果没有开启异步执行，则exec_buffer一定为0
        exec_buffer = get_forward_context().exec_buffer
        if (exec_buffer, runtime_shape) not in self.concrete_size_entries:
            # we don't need to do anything for this shape
            return self.compiled_graph_for_general_shape(*args)

        entry = self.concrete_size_entries[(exec_buffer, runtime_shape)]

        if entry.runnable is None:
            entry.runnable = self.compiled_graph_for_general_shape

        if entry.need_to_compile and not entry.compiled:

            entry.compiled = True

            self.to_be_compiled_sizes[runtime_shape]-=1

            if self.to_be_compiled_sizes[runtime_shape] == 0:
                self.to_be_compiled_sizes.pop(runtime_shape)

            entry.runnable = self.vllm_backend.compiler_manager.compile(
                self.graph,
                args,
                self.compilation_config.inductor_compile_config,
                self.compilation_config,
                graph_index=self.piecewise_compile_index,
                num_graphs=self.total_piecewise_compiles,
                runtime_shape=runtime_shape)

            # finished compilations for all required shapes
            if self.is_last_graph and not self.to_be_compiled_sizes:
                self.check_for_ending_compilation()

        # Skip CUDA graphs if this entry doesn't use them OR
        # if we're supposed to skip them globally
        skip_cuda_graphs = get_forward_context().skip_cuda_graphs

        if not entry.use_cudagraph or skip_cuda_graphs:
            return entry.runnable(*args)

        if entry.cudagraph is None:
            if entry.num_finished_warmup < self.compilation_config.cudagraph_num_of_warmups:  # noqa
                entry.num_finished_warmup += 1
                if self.is_first_graph:
                    logger.debug(
                        "Warming up %s/%s for shape %s",
                        entry.num_finished_warmup,
                        self.compilation_config.cudagraph_num_of_warmups,
                        runtime_shape)
                return entry.runnable(*args)

            if self.is_first_graph:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every shape.
                # We only log it in the debug mode.
                logger.debug("Capturing a cudagraph for shape %s",
                             runtime_shape)

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            cudagraph = torch.cuda.CUDAGraph()
            
            with ExitStack() as stack:
                if not self.is_first_graph:
                    # during every model forward, we will capture
                    # many pieces of cudagraphs (roughly one per layer).
                    # running gc again and again across layers will
                    # make the cudagraph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("torch.cuda.empty_cache", lambda: None))

                # mind-exploding: carefully manage the reference and memory.
                with torch.cuda.graph(cudagraph, pool=self.graph_pool):
                    # `output` is managed by pytorch's cudagraph pool

                    output = entry.runnable(*args)
                    if self.is_last_graph:
                        # by converting it to weak ref,
                        # the original `output` will immediately be released
                        # to save memory. It is only safe to do this for
                        # the last graph, because the output of the last graph
                        # will not be used by any other cuda graph.
                        output = weak_ref_tensors(output)

            # here we always use weak ref for the output
            # to save memory
            entry.output = weak_ref_tensors(output)
            entry.cudagraph = cudagraph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during cuda graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                "Input addresses for cudagraphs are different during replay."
                f" Expected {entry.input_addresses}, got {new_input_addresses}"
            )

        entry.cudagraph.replay()
        return entry.output
