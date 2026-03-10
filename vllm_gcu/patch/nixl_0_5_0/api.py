from unittest.mock import patch

class nixl_agent_config:
    def __init__(
        self,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = 0,
        num_threads: int = 0,
        backends: list[str] = ["UCX"],
    ):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread
        self.enable_listen = enable_listen_thread
        self.port = listen_port
        self.num_threads = num_threads

patch("nixl._api.nixl_agent_config", nixl_agent_config).start()