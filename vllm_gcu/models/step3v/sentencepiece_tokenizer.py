# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors
import glob
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import sentencepiece

from vllm.transformers_utils.tokenizer_base import TokenizerBase

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ConversationMessage


@dataclass
class Encoding:
    input_ids: List[int]


class SentencePieceTokenizer(TokenizerBase):
    """SentencePieceTokenizer"""

    def __init__(self, model_file):
        self.name = "SentencePieceTokenizer"
        self.sp_model = sentencepiece.SentencePieceProcessor(
            model_file=model_file
        )

        # Set special tokens
        self._special_tokens = {}
        self._all_special_tokens = []
        self._all_special_ids = []
        self._vocab = {}
        for idx in range(self.sp_model.get_piece_size()):
            self._vocab[self.sp_model.id_to_piece(idx)] = idx

            if not self.sp_model.is_control(idx):
                continue

            self._special_tokens[self.sp_model.id_to_piece(idx)] = idx
            self._all_special_tokens.append(self.sp_model.id_to_piece(idx))
            self._all_special_ids.append(idx)

        self._special_tokens[
            self.sp_model.id_to_piece(self.sp_model.unk_id())
        ] = self.sp_model.unk_id()
        self._all_special_tokens.append(
            self.sp_model.id_to_piece(self.sp_model.unk_id())
        )
        self._all_special_ids.append(self.sp_model.unk_id())

        # FIXME: compatible for decode
        self.length = self.sp_model.get_piece_size()

    @property
    def all_special_tokens_extended(self) -> List[str]:
        return self._all_special_tokens

    @property
    def all_special_tokens(self) -> List[str]:
        return self._all_special_tokens

    @property
    def all_special_ids(self) -> List[int]:
        return self._all_special_ids

    @property
    def eos_token_id(self):
        return self.sp_model.eos_id()

    @property
    def eos_token(self):
        return self.sp_model.id_to_piece(self.eos_token_id)

    @property
    def bos_token_id(self):
        return self.sp_model.bos_id()

    @property
    def unk_token_id(self):
        return self.sp_model.unk_id()

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def vocab_size(self):
        return self.length

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def max_token_id(self) -> int:
        return self.sp_model.get_piece_size() - 1

    def get_vocab(self):
        return self._vocab

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        # Mistral Tokenizers should not add special tokens
        input_ids = self.encode(text)

        if truncation:
            input_ids = input_ids[:max_length]
        return input_ids

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        add_bos: bool = True,
        include_control_tokens: bool = False,
    ) -> List[int]:
        if include_control_tokens:
            # encode control token as normal string
            parts = []
            current_text = text
            for token in self._special_tokens:
                while token in current_text:
                    idx = current_text.find(token)
                    if idx > 0:
                        parts.append(current_text[:idx])
                    parts.append({"token": token})
                    current_text = current_text[idx + len(token) :]

            if current_text:
                parts.append(current_text)

            return self.encode_chatml(parts, add_bos=add_bos)
        else:
            return self.sp_model.encode(text, add_bos=add_bos)

    def decode(
        self,
        token_ids: Union[List[int], int],
        skip_special_tokens: bool = False,
    ) -> str:
        if isinstance(token_ids, list) and not isinstance(token_ids[0], int):
            token_ids = [int(token) for token in token_ids]
        return self.sp_model.decode(token_ids)

    def __call__(
        self,
        text: Union[str, List[str], List[int]],
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        input_ids = self.encode(text, add_bos=True)
        if truncation:
            input_ids = input_ids[:max_length]
        return Encoding(input_ids=input_ids)

    def convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def convert_tokens_to_ids(self, tokens):
        return self.sp_model.piece_to_id(tokens)

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.id_to_piece(index)

    def convert_ids_to_tokens(
        self, ids, **kwargs
    ):  # kwargs for compatibility of HF tokenizer
        return self.sp_model.id_to_piece(ids)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    @classmethod
    def from_pretrained(cls, model_path):
        if model_path.endswith(".model"):
            model_file = model_path
        else:
            possible_files = glob.glob(f"{model_path}/*.model")
            if len(possible_files) != 1:
                raise ValueError(
                    f"Expected exactly one .model file for tokenizer initialization in {model_path}, but found {possible_files}"
                )
            model_file = possible_files[0]
        return cls(model_file=model_file)

    def encode_chatml(self, input, add_bos=True):
        input_ids = [self.bos_token_id] if add_bos else []
        if isinstance(input, str):
            input = [input]
        # Compatible with the StepChat ChatML Protocol.
        for subprompt in input:
            if isinstance(subprompt, str):
                subprompt_ids = self.encode(subprompt, add_bos=False)
                input_ids += subprompt_ids
            elif isinstance(subprompt, dict):
                if "token" in subprompt:
                    input_ids += [self.convert_token_to_id(subprompt["token"])]
        return input_ids

    def get_added_vocab(self):
        return None

    def __len__(self):
        return self.length

    def apply_chat_template(
        self,
        conversation: List["ConversationMessage"],
        tools: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[int]:
        """Convert chat messages to token IDs sequence.

        Args:
            messages: List of chat messages
            tools: Tool configurations (optional)

        Returns:
            List[int]: Sequence of token IDs
        """
        ret = [self.bos_token_id]
        continue_final_message = kwargs.get("continue_final_message", True)

        total_messages = len(conversation)
        for i, message in enumerate(conversation):
            if (
                i == total_messages - 1
                and message["role"] == "assistant"
                and not continue_final_message
            ):
                continue
            # Add BOT token
            ret.append(self._special_tokens["<|BOT|>"])

            # Process role and content
            role = "human" if message["role"] == "user" else message["role"]
            content = message["content"] or ""

            # Encode role and content
            if isinstance(content, str):
                text = f"{role}\n{content}"
                ret.extend(self.encode(text, add_bos=False))
            elif isinstance(content, list):
                text = f"{role}\n"
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        ret.extend(
                            self.encode(text + item["text"], add_bos=False)
                        )
                        text = ""
                    elif isinstance(item, dict) and item.get("type") == "image":
                        if text:
                            ret.extend(self.encode(text, add_bos=False))
                            text = ""
                        ret.append(self.special_tokens["<im_patch>"])
                    elif isinstance(item, dict) and item.get("type") == "audio":
                        if text:
                            ret.extend(self.encode(text, add_bos=False))
                            text = ""
                        ret.append(self.special_tokens["<audio_patch>"])
                    else:
                        raise ValueError(f"Unsupported item: {item}")
            else:
                raise ValueError(f"Unsupported message: {message}")

            # Add EOT token
            ret.append(self._special_tokens["<|EOT|>"])

        # If the last message is not from assistant, add assistant prompt
        if (
            conversation[-1]["role"] != "assistant"
            or not continue_final_message
        ):
            ret.append(self._special_tokens["<|BOT|>"])
            ret.extend(self.encode("assistant\n", add_bos=False))
        # If the last message is from assistant, remove the last EOT token
        elif ret[-1] == self._special_tokens["<|EOT|>"]:
            ret.pop()

        return ret
