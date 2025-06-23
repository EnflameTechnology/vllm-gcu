from contextlib import contextmanager


class CuMemAllocator:
    @staticmethod
    def get_instance():
        raise NotImplementedError

    def python_malloc_callback(self, allocation_handle) -> None:
        raise NotImplementedError

    def python_free_callback(self, ptr: int):
        raise NotImplementedError

    def sleep(self, ptr: int):
        raise NotImplementedError

    def wake_up(self, tags = None):
        raise NotImplementedError

    @contextmanager
    def use_memory_pool(self, tag = None):
        raise NotImplementedError

    def get_current_usage(self) -> int:
        raise NotImplementedError
