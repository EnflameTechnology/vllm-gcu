import sys
from unittest.mock import patch

HAS_COMPLETION_PATCHED = False
HAS_CHAT_PATCHED = False

def decorator(f):
    def inner(*args, **kwargs):
        if "request" in kwargs:
            request = kwargs["request"]
        else:
            request = args[0]

        extra_args = getattr(request, "extra_args", None)
        g = getattr(request, "to_sampling_params")

        def to_sampling_params(self, *_args, **_kwargs):
            sampling_params = g(*_args, **_kwargs)
            if extra_args:
                if sampling_params.extra_args is None:
                    sampling_params.extra_args = extra_args
                else:
                    sampling_params.extra_args.update(extra_args)
            return sampling_params

        setattr(request, "to_sampling_params", to_sampling_params.__get__(request))

        if "request" in kwargs:
            kwargs["request"] = request
        else:
            args = (request,) + args[1:]

        return f(*args, **kwargs)

    return inner


def completion(request):
    global HAS_COMPLETION_PATCHED
    if not HAS_COMPLETION_PATCHED:
        g = getattr(request.app.state.openai_serving_completion, "create_completion")
        setattr(
            request.app.state.openai_serving_completion,
            "create_completion",
            decorator(g),
        )
        HAS_COMPLETION_PATCHED = True

    return request.app.state.openai_serving_completion


def chat(request):
    global HAS_CHAT_PATCHED
    if not HAS_CHAT_PATCHED:
        g = getattr(request.app.state.openai_serving_chat, "create_chat_completion")
        setattr(
            request.app.state.openai_serving_chat,
            "create_chat_completion",
            decorator(g),
        )
        HAS_CHAT_PATCHED = True

    return request.app.state.openai_serving_chat


if sys.modules["__main__"].__package__ == "vllm.entrypoints.openai":
    setattr(sys.modules["__main__"], "completion", completion)
    setattr(sys.modules["__main__"], "chat", chat)
