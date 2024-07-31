import dataclasses
import inspect
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union

from outlines.generate.api import GenerationParameters, SamplingParameters

from .tokenizer import OutlinesTokenizer

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


class Transformers:
    """Represents a `transformers` model."""

    def __init__(
        self,
        model: "PreTrainedModel",
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.outlines_tokenizer = OutlinesTokenizer.from_tokenizer(tokenizer)

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

    def generate(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Union[str, List[str], List[List[str]]]:
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
        inputs = self.tokenizer.batch_encode_plus(
            [prompts] if isinstance(prompts, str) else prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        """
        Temporary stream stand-in which implements stream() signature
        and equivalent behaviour but isn't yielded until generation completes.

        TODO: implement following completion of https://github.com/huggingface/transformers/issues/30810
        """
        inputs = self.tokenizer.batch_encode_plus(
            [prompts] if isinstance(prompts, str) else prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            generated_ids = generated_ids.squeeze(0)

        for i in range(generated_ids.size(-1)):
            output_group_ids = generated_ids.select(-1, i).unsqueeze(-1)
            yield self._decode_generation(output_group_ids)

    def _get_generation_kwargs(
        self,
        prompts: Union[str, List[str]],
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> dict:
        """
        Conert outlines generation parameters into model.generate kwargs
        """
        from transformers import GenerationConfig, LogitsProcessorList, set_seed

        max_new_tokens, stop_at, seed = dataclasses.astuple(generation_parameters)
        sampler, num_samples, top_p, top_k, temperature = dataclasses.astuple(
            sampling_parameters
        )
        if max_new_tokens is None:
            max_new_tokens = int(2**30)

        # global seed, not desirable
        if seed is not None:
            set_seed(seed)

        if logits_processor is not None:
            logits_processor_list = LogitsProcessorList([logits_processor])
        else:
            logits_processor_list = None

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            stop_strings=stop_at,
            num_return_sequences=(num_samples or 1),
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            do_sample=(sampler == "multinomial"),
            num_beams=(num_samples if sampler == "beam_search" else 1),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return dict(
            logits_processor=logits_processor_list,
            generation_config=generation_config,
            tokenizer=self.tokenizer,
        )

    def _generate_output_seq(
        self, prompts, inputs, generation_config, **generation_kwargs
    ):
        input_ids = inputs["input_ids"]
        output_ids = self.model.generate(
            **inputs, generation_config=generation_config, **generation_kwargs
        )

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if self.model.config.is_encoder_decoder:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
        num_samples = generation_config.num_return_sequences or 1

        if num_samples > 1 and isinstance(prompts, list):
            batch_size = input_ids.size(0)
            num_return_sequences = generation_config.num_return_sequences or 1
            generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)

        return generated_ids

    def _decode_generation(self, generated_ids: "torch.Tensor"):
        if len(generated_ids.shape) == 1:
            return self.tokenizer.batch_decode(
                [generated_ids], skip_special_tokens=True
            )[0]
        elif len(generated_ids.shape) == 2:
            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        elif len(generated_ids.shape) == 3:
            return [
                self.tokenizer.batch_decode(generated_ids[i], skip_special_tokens=True)
                for i in range(len(generated_ids))
            ]
        else:
            raise TypeError(
                f"Generated outputs aren't 1D, 2D or 3D, but instead are {generated_ids.shape}"
            )


def transformers(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
    model_class=None,
    tokenizer_class=None,
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
    if model_class is None or tokenizer_class is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if model_class is None:
        model_class = AutoModelForCausalLM
    if tokenizer_class is None:
        tokenizer_class = AutoTokenizer

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    tokenizer_kwargs.setdefault("padding_side", "left")
    tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return Transformers(model, tokenizer)


def mamba(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    tokenizer_kwargs: dict = {},
):
    try:
        from transformers import MambaForCausalLM

    except ImportError:
        raise ImportError(
            "The `mamba_ssm`, `torch` and `transformer` libraries needs to be installed in order to use Mamba."
        )

    return transformers(
        model_name=model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        model_class=MambaForCausalLM,
    )
