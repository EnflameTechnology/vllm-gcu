from unittest.mock import patch


def tops_device_count():
    import torch_gcu  # noqa

    return torch_gcu._C._gcu_getDeviceCount()

patch("vllm.utils.cuda_device_count_stateless", tops_device_count).start()
