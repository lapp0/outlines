import pickle
from typing import (
    TYPE_CHECKING,
    Dict,
    Hashable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
from numpy.typing import NDArray

if TYPE_CHECKING:
    from llama_cpp.llama_tokenizer import LlamaHFTokenizer, LlamaTokenizer
    from mlxlm.tokenizer_utils import TokenizerWrapper
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from vllm.transformers_utils.tokenizer_group import BaseTokenizerGroup

    ConvertableTokenizer = Union[
        PreTrainedTokenizerBase,
        LlamaTokenizer,
        LlamaHFTokenizer,
        BaseTokenizerGroup,
        TokenizerWrapper,
    ]


class OutlinesTokenizer:
    """
    Immutable class.

    Minimal tokenizer necessary for building an automata for the base tokenizer.

    Strictly to be used for structured generation.
    """

    vocabulary: Dict[str, int]
    normalized_vocabulary: Dict[str, str]
    special_tokens: Set[str]
    eos_token_id: int
    eos_token: str
    pad_token_id: int
    pad_token: str

    def __init__(
        self,
        vocabulary: Dict[str, int],
        normalized_vocabulary: Dict[str, str],
        special_tokens: Set[str],
        eos_token: str,
        pad_token: Optional[str],
    ):
        if pad_token is None:
            pad_token = eos_token

        self.vocabulary = vocabulary
        self.normalized_vocabulary = normalized_vocabulary
        self.special_tokens = special_tokens
        self.eos_token = eos_token
        self.eos_token_id = self.vocabulary[eos_token]
        self.pad_token = pad_token
        self.pad_token_id = self.vocabulary[pad_token]

        self._hash = None  # only mutable attribute, cached hash

    @classmethod
    def from_tokenizer(
        cls, tokenizer: Union["OutlinesTokenizer", "ConvertableTokenizer"]
    ):
        """
        Create an OutlinesTokenizer based on the type of the ConvertableTokenizer
        """
        import importlib

        tokenizer_mapping = {
            "outlines.models.tokenizer.OutlinesTokenizer": lambda t: t,
            "transformers.tokenization_utils_base.PreTrainedTokenizerBase": cls.from_hf_tokenizer,
            "llama_cpp.llama_tokenizer.LlamaTokenizer": cls.from_llamacpp_tokenizer,
            "llama_cpp.llama_tokenizer.LlamaHFTokenizer": lambda t: cls.from_df_tokenizer(
                t.hf_tokenizer
            ),
            "vllm.transformers_utils.tokenizer_group.BaseTokenizerGroup": lambda t: cls.from_df_tokenizer(
                t.tokenizer
            ),
            "mlxlm.tokenizer_utils.TokenizerWrapper": lambda t: cls.from_hf_tokenizer(
                t.tokenizer
            ),
        }
        for module_class, handler in tokenizer_mapping.items():
            module_name, class_name = module_class.rsplit(".", 1)
            module_spec = importlib.util.find_spec(module_name)
            if module_spec:
                module = importlib.import_module(module_name)
                cls_ = getattr(module, class_name, None)
                if cls_ and isinstance(tokenizer, cls_):
                    return handler(tokenizer)

        raise TypeError("Expected PR TODO FULL MESSAGE")

    @classmethod
    def from_hf_tokenizer(cls, tokenizer: "PreTrainedTokenizerBase"):
        """
                PR TODO: WERE MISSING

        -        from transformers.file_utils import SPIECE_UNDERLINE
        -
        -        string = self.tokenizer.convert_tokens_to_string([token])
        -
        -        if self.is_llama:
        -            # A hack to handle missing spaces to HF's Llama tokenizers
        -            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
        -                return " " + string
        -
        -        return string

        """
        normalized_vocabulary = {
            token: tokenizer.convert_tokens_to_string([token])
            for token in tokenizer.get_vocab()
        }
        return cls(
            vocabulary=tokenizer.get_vocab(),
            normalized_vocabulary=normalized_vocabulary,
            special_tokens=set(tokenizer.all_special_tokens),
            eos_token=tokenizer.eos_token,
            pad_token=tokenizer.pad_token,
        )

    @classmethod
    def from_llamacpp_tokenizer(cls, tokenizer: "LlamaTokenizer"):
        # use tokenizer._model
        import pdb

        pdb.set_trace()
        raise Exception()
        return cls(
            vocabulary=TODO,
            normalized_vocabulary=TOOD,
            special_tokens=TODO,
            eos_token=tokenizer.decode([tokenizer._model.token_eos()]),
            pad_token=tokenizer.decode([tokenizer._model.token_pad()]),
        )

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        return self.normalized_vocabulary[token]

    def __getstate__(self):
        """Create a stable representation for outlines.caching"""
        return (
            tuple(sorted(self.vocabulary.items())),
            tuple(sorted(self.normalized_vocabulary.items())),
            tuple(sorted(self.special_tokens)),
            self.eos_token,
            self.eos_token_id,
            self.pad_token,
            self.pad_token_id,
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__getstate__() == other.__getstate__()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.__getstate__())
        return self._hash
