from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

KVConnectorFactory.register_connector(
    "P2pEcclConnector",
    "vllm_gcu.distributed.kv_transfer.v1.p2p_eccl_connector",
    "P2pEcclConnector")

KVConnectorFactory.register_connector(
    "NixlLayerwiseConnector",
    "vllm_gcu.distributed.kv_transfer.v1.nixl_layerwise_connector",
    "NixlLayerwiseConnector",
)

KVConnectorFactory.register_connector(
    "DecodeBenchConnector",
    "vllm_gcu.distributed.kv_transfer.v1.decode_bench_connector",
    "DecodeBenchConnector",
)
