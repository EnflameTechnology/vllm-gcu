from collections import UserDict, defaultdict

from vllm.multimodal.inputs import NestedTensors

def gather_kwargs(features: list["MultiModalFeatureSpec"], keys: set[str]):
    kwargs = defaultdict[str, list[NestedTensors]](list)

    for f in features:
        item = f.data
        if item is not None:
            for k in keys:
                if k in item:
                    kwargs[k].append(item[k].data)

    return dict(kwargs)