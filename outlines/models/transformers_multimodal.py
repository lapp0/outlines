from outlines.models import Transformers
from outlines.generate.api import GenerationParameters, SamplingParameters

from typing import TYPE_CHECKING, List, Optional, Union, Any, Iterator

if TYPE_CHECKING:
    from outlines.processors import OutlinesLogitsProcessor


class TransformersMultiModal(Transformers):
    def __init__(self, model, tokenizer, processor):
        super().__init__(model, tokenizer)
        self.processor = processor

    def generate(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[Any, List[Any]],  # TODO: docstring
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
        if isinstance(prompts, str):
            inputs = self.processor(
                prompts,
                media,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

        else:
            raise NotImplementedError("TODO")

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            # Should always be true until NotImplementedError above is fixed
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def stream(  # type: ignore
        self,
        prompts: Union[str, List[str]],
        media: Union[Any, List[Any]],  # TODO: docstring
        generation_parameters: GenerationParameters,
        logits_processor: Optional["OutlinesLogitsProcessor"],
        sampling_parameters: SamplingParameters,
    ) -> Iterator[Union[str, List[str]]]:
        """
        Temporary stream stand-in which implements stream() signature
        and equivalent behaviour but isn't iterable.
        """
        if isinstance(prompts, str):
            inputs = self.processor(
                prompts,
                media,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

        else:
            raise NotImplementedError("TODO")

        generation_kwargs = self._get_generation_kwargs(
            prompts,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(prompts, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(prompts, str):
            # Should always be true until NotImplementedError above is fixed
            generated_ids = generated_ids.squeeze(0)

        for i in range(generated_ids.size(-1)):
            output_group_ids = generated_ids.select(-1, i).unsqueeze(-1)
            yield self._decode_generation(output_group_ids)


def transformers_multimodal(
    model_name: str,
    device: Optional[str] = None,
    model_kwargs: dict = {},
    processor_kwargs: dict = {},
    model_class=None,
    tokenizer_class=None,
    processor_class=None,
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
    processor_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the processor.

    Returns
    -------
    A `TransformersModel` model instance.

    """
    if model_class is None or tokenizer_class is None:
        try:
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The `transformers` library needs to be installed in order to use `transformers` models."
            )
    if model_class is None:
        model_class = LlavaNextForConditionalGeneration
    if processor_class is None:
        processor_class = LlavaNextProcessor

    if device is not None:
        model_kwargs["device_map"] = device

    model = model_class.from_pretrained(model_name, **model_kwargs)

    processor_kwargs.setdefault("padding_side", "left")
    processor_kwargs.setdefault("pad_token", "[PAD]")
    processor = processor_class.from_pretrained(model_name, **processor_kwargs)

    if tokenizer_class is None:
        if getattr(processor, "tokenizer", None):
            tokenizer = processor.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **processor_kwargs)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name, **processor_kwargs)

    return TransformersMultiModal(model, tokenizer, processor)
