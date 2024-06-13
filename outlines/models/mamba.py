from typing import Optional

from .transformers import transformers


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
            "The `mamba_ssm`, `torch` and `transformer` libraries needs to be installed in order to use Mamba people."
        )

    return transformers(
        model_name=model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        model_class=MambaForCausalLM,
    )
