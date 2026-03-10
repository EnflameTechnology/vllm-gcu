import os
import warnings
from vllm import envs
import huggingface_hub
from pathlib import Path
from vllm.logger import init_logger
from typing import TYPE_CHECKING, Any, Optional, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from vllm.transformers_utils.tokenizers import MistralTokenizer
from vllm.transformers_utils.utils import check_gguf_file
from vllm.transformers_utils.config import (
    get_sentence_transformer_tokenizer_config)
from vllm.transformers_utils.tokenizer import get_cached_tokenizer

if TYPE_CHECKING:
    from vllm.transformers_utils.tokenizer_base import TokenizerBase
else:
    TokenizerBase = Any

from vllm_gcu.tokenizers import DeepseekV32Tokenizer
from unittest.mock import patch

logger = init_logger(__name__)

AnyTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast,
                     TokenizerBase]


def get_tokenizer(
    tokenizer_name: Union[str, Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope.
    """
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circuit import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=tokenizer_name,
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])
                tokenizer_name = tokenizer_path

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    # if tokenizer is from official mistral org
    is_from_mistral_org = str(tokenizer_name).split("/")[0] == "mistralai"
    if is_from_mistral_org and tokenizer_mode != "mistral":
        warnings.warn(
            'It is strongly recommended to run mistral models with '
            '`--tokenizer-mode "mistral"` to ensure correct '
            'encoding and decoding.',
            FutureWarning,
            stacklevel=2)

    tokenizer: AnyTokenizer
    if tokenizer_mode == "mistral":
        tokenizer = MistralTokenizer.from_pretrained(str(tokenizer_name),
                                                     revision=revision)
    elif tokenizer_mode == "deepseek_v32":
        tokenizer = DeepseekV32Tokenizer.from_pretrained(str(tokenizer_name),
                                                     revision=revision)
    elif tokenizer_mode == "custom":
        from vllm.transformers_utils.tokenizer_base import TokenizerRegistry
        tokenizer = TokenizerRegistry.get_tokenizer(str(tokenizer_name),
                                                    *args,
                                                    revision=revision,
                                                    download_dir=download_dir,
                                                    **kwargs)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                    "does not exist or is not currently imported." in str(e)
                    or "requires you to execute the tokenizer file" in str(e)):
                err_msg = ("Failed to load the tokenizer. If the tokenizer "
                           "is a custom tokenizer not yet available in the "
                           "HuggingFace transformers library, consider "
                           "setting `trust_remote_code=True` in LLM or using "
                           "the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            tokenizer_name, revision)
        if isinstance(encoder_config, dict) and encoder_config.get(
                "do_lower_case", False):
            special_tokens_map = {
                k: v.lower()
                for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead.")
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer


patch("vllm.transformers_utils.tokenizer.get_tokenizer", get_tokenizer).start()
