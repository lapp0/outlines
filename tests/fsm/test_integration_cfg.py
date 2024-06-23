from outlines.fsm.guide import CFGGuide, Generate
import outlines.grammars as grammars
import outlines.models as models

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tokenizer_bpe():
    return models.transformers(
        "hf-internal-testing/tiny-random-gpt2", device="cpu"
    ).tokenizer


@pytest.fixture(scope="session")
def tokenizer_character_level():
    return models.transformers("google/byt5-small", device="cpu").tokenizer


TOKENIZERS = ["tokenizer_bpe"]  # , "tokenizer_character_level"]


# Collects all samples within cfg_samples/ and makes adding
# a test case as easy as adding a valid sample to cfg_samples/
all_samples = {}
examples_path = Path(__file__).parent.parent / "cfg_samples"
for sample_collection_path in examples_path.iterdir():
    grammar_name = sample_collection_path.name
    grammar = getattr(grammars, grammar_name)
    for sample_path in sample_collection_path.iterdir():
        test_name = f"{grammar_name}_{sample_path.name}"
        with open(sample_path) as f:
            all_samples[test_name] = (grammar, f.read().rstrip("\n"))


@pytest.mark.parametrize("sample_name", all_samples.keys())
@pytest.mark.parametrize("tokenizer_name", TOKENIZERS)
def test_cfg_sample_valid(request, sample_name, tokenizer_name):
    tokenizer = request.getfixturevalue(tokenizer_name)

    cfg, sample = all_samples[sample_name]
    cfg_guide = CFGGuide(cfg, tokenizer)

    # TODO: assert that the sample is valid under the grammar using
    # pure lark, if its not raise an appropriate exception

    sample_token_ids = tokenizer.encode(sample)[0][0]
    assert len(sample_token_ids.shape) == 1  # ensure we're encoding in the desired shape for this test

    state = 0
    for i, token_id in enumerate(sample_token_ids):
        next_instruction = cfg_guide.get_next_instruction(state)
        if token_id not in next_instruction.tokens:
            processed_str = tokenizer.decode([sample_token_ids[:i]])[0]
            remaining_str = tokenizer.decode([sample_token_ids[i:]])[0]
            if next_instruction.tokens == [tokenizer.eos_token_id]:
                error_label = "CFGGuide required EOS early"
            else:
                expected = tokenizer.decode(next_instruction.tokens)
                error_label = f"Mismatched expectations, Guide expected {expected}"
            raise Exception(
                f"{error_label}\n"
                f"processed:\n```{processed_str}```\n"
                f"remaining:\n```{remaining_str}```"
            )
            next_instruction.tokens
        state = cfg_guide.get_next_state(state, token_id)

    final_instruction = cfg_guide.get_next_instruction(state)
    assert tokenizer.eos_token_id in final_instruction
