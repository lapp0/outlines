import dataclasses
from threading import Thread
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

from datasets.fingerprint import Hasher

from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.models.tokenizer import Tokenizer

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from outlines.processors import OutlinesLogitsProcessor

__all__ = ["transformers"]


KVCacheType = Tuple[Tuple["torch.DoubleTensor", "torch.DoubleTensor"], ...]


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: "PreTrainedTokenizer", **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple["torch.LongTensor", "torch.LongTensor"]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: "torch.LongTensor") -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)

    def forward(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple["torch.FloatTensor", Optional[KVCacheType]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        try:
            import torch
        except ImportError:
            ImportError(
                "The `torch` library needs to be installed to use `transformers` models."
            )
        assert 0 < input_ids.ndim < 3

        if past_key_values:
            input_ids = input_ids[..., -1].unsqueeze(-1)

        with torch.inference_mode():
            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                past_key_values=past_key_values,
            )

        return output.logits, output.past_key_values

    def __call__(
        self,
        input_ids: "torch.LongTensor",
        attention_mask: "torch.LongTensor",
        past_key_values: Optional[Tuple] = None,
    ) -> "torch.FloatTensor":
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, kv_cache

    def _make_generate_kwargs(
        self,
        generation_parameters: GenerationParameters,
        sampling_parameters: SamplingParameters,
    ) -> dict:
        """
        Conert outlines generation parameters into the **kwargs dict passed
        to model.generate()
        """
        max_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )

        if sampler != "multinomial":
            if top_k is not None:
                raise ValueError(f"{sampler} requires top_k to be None")
            if top_p is not None:
                raise ValueError(f"{sampler} requires top_p to be None")

        if sampler == "greedy":
            if num_samples is not None and num_samples != 1:
                raise ValueError(f"{sampler} requires num_samples to be 1")
            if temperature is not None and temperature != 0:
                raise ValueError(f"{sampler} requires temperature to be 0 or None")

        if isinstance(stop_at, str):
            stop_at = [stop_at]

        generate_kwargs = dict(
            max_new_tokens=max_tokens,
            stop_strings=stop_at,
            seed=seed,
            num_return_sequences=(num_samples or 1),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )

        if sampler == "multinomial":
            generate_kwargs["do_sample"] = True

        elif sampler == "beam_search":
            generate_kwargs["num_beams"] = num_samples

        elif sampler == "greedy":
            pass

        else:
            raise TypeError(f"Incompatible Sampler: {sampler}")

        return generate_kwargs

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str]]:
        """Generate text using `transformers`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        The generated text
        """
        from transformers import LogitsProcessorList

        if isinstance(prompts, str):
            # convert to 2d
            input_ids, attention_mask = self.tokenizer.encode([prompts])
        else:
            input_ids, attention_mask = self.tokenizer.encode(prompts)

        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        generate_kwargs = self._make_generate_kwargs(
            generation_parameters,
            sampling_parameters,
        )

        if logits_processor is not None:
            logits_processor_list = LogitsProcessorList([logits_processor])
        else:
            logits_processor_list = None

        full_seq_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor_list,
            **generate_kwargs,
        )
        generated_ids = full_seq_ids[:, input_ids.shape[1] :]

        outputs = self.tokenizer.decode(generated_ids)

        if isinstance(prompts, str):
            # convert back to 1d
            return outputs[0]
        else:
            return outputs

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[str]:
        """Stream text using `transformers`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.

        Returns
        -------
        A token generator
        """
        from transformers import LogitsProcessorList, TextIteratorStreamer

        if not isinstance(prompts, str):
            raise TypeError("Cannot stream batch inputs")

        input_ids, attention_mask = self.tokenizer.encode(prompts)
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        generate_kwargs = self._make_generate_kwargs(
            generation_parameters,
            sampling_parameters,
        )

        if logits_processor is not None:
            logits_processor_list = LogitsProcessorList([logits_processor])
        else:
            logits_processor_list = None

        streamer = TextIteratorStreamer(self.tokenizer.tokenizer)
        kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_processor=logits_processor_list,
            streamer=streamer,
            **generate_kwargs,
        )
        thread = Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    """Instantiate a model from the `transformers` library and its tokenizer.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    device
        The device(s) on which the model should be loaded. This overrides
        the `device_map` entry in `model_kwargs` when provided.
    model_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the model.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "The `transformers` library needs to be installed in order to use `transformers` models."
        )

    if device is not None:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    return Transformers(model, tokenizer)
