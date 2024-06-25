from functools import singledispatch

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import MLXLM, OpenAI
from outlines.processors import OutlinesLogitsProcessor
from outlines.samplers import Sampler, multinomial


@singledispatch
def base(
    model, logits_processor: OutlinesLogitsProcessor, sampler: Sampler = multinomial()
):
    """Generate structured text directly using OutlinesLogitsProcessor's

    Parameters
    ----------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    logits_processor:
        The logits processor applied after each decoder pass.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGenerator` instance that generates text constrained by the
    regular expression.

    """
    raise NotImplementedError


@base.register(MLXLM)
def base_unified(
    model: MLXLM,
    logits_processor: OutlinesLogitsProcessor,
    sampler: Sampler = multinomial(),
):
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@base.register(OpenAI)
def base_openai(
    model: OpenAI,
    logits_processor: OutlinesLogitsProcessor,
    sampler: Sampler = multinomial(),
) -> Exception:
    raise NotImplementedError("The OpenAI API does not support logits processing.")
