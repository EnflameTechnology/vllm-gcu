from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


KVConnectorFactory.register_connector(
    "PyEcclConnector",
    "vllm_gcu.distributed.kv_transfer.connector",
    "Connector",
)

KVConnectorFactory.register_connector(
    "P2pEcclConnector",
    "vllm_gcu.distributed.kv_transfer.v1.p2p_eccl_connector",
    "P2pEcclConnector")