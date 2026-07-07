import pytest
import types
from unittest.mock import patch

import vllm_gcu.kernels.prepare_finalize as pf


@pytest.fixture(autouse=True)
def mock_ep_group(monkeypatch):
    class FakeDeviceGroup:
        def size(self):
            return 1

    class FakeEPGroup:
        device_group = FakeDeviceGroup()

    monkeypatch.setattr(pf, "get_ep_group", lambda: FakeEPGroup())


def test_num_dispatchers():
    num_dispatchers = 4

    impl = pf.AlltoAllStaticShape(
        threshold=1024,
        num_dispatchers=num_dispatchers,
    )

    assert impl.num_dispatchers() == num_dispatchers


def test_static_max_num_tokens_per_rank():
    threshold = 8192
    num_dispatchers = 2

    impl = pf.AlltoAllStaticShape(
        threshold=threshold,
        num_dispatchers=num_dispatchers,
    )

    expected = threshold // num_dispatchers
    assert impl.max_num_tokens_per_rank() == expected


def test_dynamic_max_num_tokens_per_rank():
    impl = pf.AlltoAllDynamicShape(num_dispatchers=4)

    assert impl.max_num_tokens_per_rank() is None


def test_selector_max_num_tokens_per_rank():
    impl = pf.AlltoAllSelector(
        threshold=4096,
        num_dispatchers=2,
    )

    assert impl.max_num_tokens_per_rank() is None
