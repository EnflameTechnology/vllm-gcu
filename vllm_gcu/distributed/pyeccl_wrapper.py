import ctypes
import platform
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.distributed import ReduceOp

from vllm.logger import init_logger

logger = init_logger(__name__)

# === export types and functions from eccl to Python ===
# for the original eccl definition, please check
# https://github.com/NVIDIA/eccl/blob/master/src/eccl.h.in

ecclResult_t = ctypes.c_int
ecclComm_t = ctypes.c_void_p


class ecclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


topsStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ecclDataType_t = ctypes.c_int


class ecclDataTypeEnum:
    ecclInt8 = 0
    ecclChar = 0
    ecclUint8 = 1
    ecclInt32 = 2
    ecclInt = 2
    ecclUint32 = 3
    ecclInt64 = 4
    ecclUint64 = 5
    ecclFloat16 = 6
    ecclHalf = 6
    ecclFloat32 = 7
    ecclFloat = 7
    ecclFloat64 = 8
    ecclDouble = 8
    ecclBfloat16 = 9
    ecclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ecclInt8
        if dtype == torch.uint8:
            return cls.ecclUint8
        if dtype == torch.int32:
            return cls.ecclInt32
        if dtype == torch.int64:
            return cls.ecclInt64
        if dtype == torch.float16:
            return cls.ecclFloat16
        if dtype == torch.float32:
            return cls.ecclFloat32
        if dtype == torch.float64:
            return cls.ecclFloat64
        if dtype == torch.bfloat16:
            return cls.ecclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ecclRedOp_t = ctypes.c_int


class ecclRedOpTypeEnum:
    ecclSum = 0
    ecclProd = 1
    ecclMax = 2
    ecclMin = 3
    ecclAvg = 4
    ecclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ecclSum
        if op == ReduceOp.PRODUCT:
            return cls.ecclProd
        if op == ReduceOp.MAX:
            return cls.ecclMax
        if op == ReduceOp.MIN:
            return cls.ecclMin
        if op == ReduceOp.AVG:
            return cls.ecclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class ECCLLibrary:
    exported_functions = [
        # const char* ecclGetErrorString(ecclResult_t result)
        Function("ecclGetErrorString", ctypes.c_char_p, [ecclResult_t]),
        # ecclResult_t  ecclGetVersion(int *version);
        Function("ecclGetVersion", ecclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        # ecclResult_t ecclGetUniqueId(ecclUniqueId* uniqueId);
        Function("ecclGetUniqueId", ecclResult_t, [ctypes.POINTER(ecclUniqueId)]),
        # ecclResult_t  ecclCommInitRank(
        #   ecclComm_t* comm, int nranks, ecclUniqueId commId, int rank);
        # note that ecclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function(
            "ecclCommInitRank",
            ecclResult_t,
            [ctypes.POINTER(ecclComm_t), ctypes.c_int, ecclUniqueId, ctypes.c_int],
        ),
        # ecclResult_t  ecclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ecclDataType_t datatype, ecclRedOp_t op, ecclComm_t comm,
        #   topsStream_t stream);
        # note that topsStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ecclAllReduce",
            ecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ecclRedOp_t,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # ecclResult_t  ecclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ecclDataType_t datatype, ecclComm_t comm,
        #   topsStream_t stream);
        # note that topsStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ecclAllGather",
            ecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # ecclResult_t  ecclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ecclDataType_t datatype, ecclRedOp_t op, ecclComm_t comm,
        #   topsStream_t stream);
        # note that topsStream_t is a pointer type, so the last argument
        # is a pointer
        Function(
            "ecclReduceScatter",
            ecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ecclRedOp_t,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # ecclResult_t  ecclSend(
        #   const void* sendbuff, size_t count, ecclDataType_t datatype,
        #   int dest, ecclComm_t comm, topsStream_t stream);
        Function(
            "ecclSend",
            ecclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ctypes.c_int,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # ecclResult_t  ecclRecv(
        #   void* recvbuff, size_t count, ecclDataType_t datatype,
        #   int src, ecclComm_t comm, topsStream_t stream);
        Function(
            "ecclRecv",
            ecclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ctypes.c_int,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # ecclResult_t ecclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   ecclDataType_t datatype, int root, ecclComm_t comm,
        #   topsStream_t stream);
        Function(
            "ecclBroadcast",
            ecclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ecclDataType_t,
                ctypes.c_int,
                ecclComm_t,
                topsStream_t,
            ],
        ),
        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # ecclResult_t  ecclCommDestroy(ecclComm_t comm);
        Function("ecclCommDestroy", ecclResult_t, [ecclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):

        so_file = so_file or "/usr/lib/libeccl.so"

        try:
            if so_file not in ECCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                ECCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = ECCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load ECCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the eccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable VLLM_ECCL_SO_PATH"
                " to point to the correct eccl library path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in ECCLLibrary.path_to_dict_mapping:
            _funcs: Dict[str, Any] = {}
            for func in ECCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            ECCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = ECCLLibrary.path_to_dict_mapping[so_file]

    def ecclGetErrorString(self, result: ecclResult_t) -> str:
        return self._funcs["ecclGetErrorString"](result).decode("utf-8")

    def ECCL_CHECK(self, result: ecclResult_t) -> None:
        if result != 0:
            error_str = self.ecclGetErrorString(result)
            raise RuntimeError(f"ECCL error: {error_str}")

    def ecclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.ECCL_CHECK(self._funcs["ecclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ecclGetUniqueId(self) -> ecclUniqueId:
        unique_id = ecclUniqueId()
        self.ECCL_CHECK(self._funcs["ecclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ecclCommInitRank(
        self, world_size: int, unique_id: ecclUniqueId, rank: int
    ) -> ecclComm_t:
        comm = ecclComm_t()
        self.ECCL_CHECK(
            self._funcs["ecclCommInitRank"](
                ctypes.byref(comm), world_size, unique_id, rank
            )
        )
        return comm

    def ecclAllReduce(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        # `datatype` actually should be `ecclDataType_t`
        # and `op` should be `ecclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ECCL_CHECK(
            self._funcs["ecclAllReduce"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ecclReduceScatter(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        op: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        # `datatype` actually should be `ecclDataType_t`
        # and `op` should be `ecclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ECCL_CHECK(
            self._funcs["ecclReduceScatter"](
                sendbuff, recvbuff, count, datatype, op, comm, stream
            )
        )

    def ecclAllGather(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        # `datatype` actually should be `ecclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.ECCL_CHECK(
            self._funcs["ecclAllGather"](
                sendbuff, recvbuff, count, datatype, comm, stream
            )
        )

    def ecclSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        self.ECCL_CHECK(
            self._funcs["ecclSend"](sendbuff, count, datatype, dest, comm, stream)
        )

    def ecclRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        self.ECCL_CHECK(
            self._funcs["ecclRecv"](recvbuff, count, datatype, src, comm, stream)
        )

    def ecclBroadcast(
        self,
        sendbuff: buffer_type,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        root: int,
        comm: ecclComm_t,
        stream: topsStream_t,
    ) -> None:
        self.ECCL_CHECK(
            self._funcs["ecclBroadcast"](
                sendbuff, recvbuff, count, datatype, root, comm, stream
            )
        )

    def ecclCommDestroy(self, comm: ecclComm_t) -> None:
        self.ECCL_CHECK(self._funcs["ecclCommDestroy"](comm))


__all__ = [
    "ECCLLibrary",
    "ecclDataTypeEnum",
    "ecclRedOpTypeEnum",
    "ecclUniqueId",
    "ecclComm_t",
    "topsStream_t",
    "buffer_type",
]
