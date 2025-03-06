from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


KVConnectorFactory.register_connector(
    "PyEcclConnector",
    "vllm_gcu.distributed.kv_transfer.connector",
    "Connector",
)
