from functools import singledispatch

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.models import MLXLM, OpenAI
from outlines.processors import OutlinesLogitsProcessor
from outlines.samplers import Sampler, multinomial


@singledispatch
def base(
    model, logits_processor: OutlinesLogitsProcessor, sampler: Sampler = multinomial()
):
    return SequenceGeneratorAdapter(model, logits_processor, sampler)


@base.register(OpenAI)
def base_openai(
    model: OpenAI,
    logits_processor: OutlinesLogitsProcessor,
    sampler: Sampler = multinomial(),
) -> Exception:
    raise NotImplementedError("The OpenAI API does not support logits processing.")
