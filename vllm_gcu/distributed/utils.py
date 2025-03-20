#!/usr/bin/env python
# coding=utf-8
import torch
import torch_gcu  # noqa: F401
from torch.distributed import ProcessGroup, is_eccl_available
from torch.distributed.distributed_c10d import (
    _get_default_timeout,
    Backend,
    PrefixStore,
)
from torch.distributed.rendezvous import rendezvous


def stateless_init_torch_distributed_process_group(
    host: str, port: int, rank: int, world_size: int, backend: str
) -> ProcessGroup:
    init_method = f"tcp://{host}:{port}"
    backend = Backend(backend)  # it is basically string
    timeout = _get_default_timeout(backend)

    store, rank, world_size = next(
        rendezvous(init_method, rank, world_size, timeout=timeout)
    )
    store.set_timeout(timeout)

    group_rank = rank
    group_size = world_size

    # Use a PrefixStore to avoid accidental overrides of keys used by
    # different systems (e.g. RPC) in case the store is multi-tenant.
    prefix_store = PrefixStore(init_method, store)

    pg: ProcessGroup = ProcessGroup(
        prefix_store,
        group_rank,
        group_size,
        ProcessGroup.Options(
            backend=backend,
        ),
    )

    if backend == "gloo":
        from torch.distributed.distributed_c10d import ProcessGroupGloo

        backend_class = ProcessGroupGloo(
            prefix_store, group_rank, group_size, timeout=timeout
        )
        backend_type = ProcessGroup.BackendType.GLOO
        device = torch.device("cpu")
    elif backend == "eccl":
        assert is_eccl_available()
        from torch_gcu._C._distributed_c10d import ProcessGroupECCL

        backend_options = ProcessGroupECCL.Options()
        backend_options._timeout = timeout

        backend_class = ProcessGroupECCL(
            prefix_store, group_rank, group_size, backend_options
        )
        backend_type = ProcessGroup.BackendType.ECCL  # not impl
        device = torch.device("gcu")
    else:
        raise RuntimeError(f"Unsupported torch distributed backend: {backend}")

    backend_class._set_sequence_number_for_group()

    pg._register_backend(device, backend_type, backend_class)

    return pg
