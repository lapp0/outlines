import contextlib
import re

import pytest

import outlines.generate as generate
import outlines.models as models
import outlines.samplers as samplers


@pytest.fixture(scope="session")
def model_llamacpp(tmp_path_factory):
    return models.llamacpp(
        repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
        filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf",
    )


@pytest.fixture(scope="session")
def model_mlxlm(tmp_path_factory):
    return models.mlxlm("mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit")


@pytest.fixture(scope="session")
def model_transformers(tmp_path_factory):
    return models.transformers("Locutusque/TinyMistral-248M-v2-Instruct", device="cpu")


@pytest.fixture(scope="session")
def model_vllm(tmp_path_factory):
    return models.vllm("facebook/opt-125m")


# TODO: mamba / exllamav2 failing in main, address in https://github.com/outlines-dev/outlines/issues/808
"""
@pytest.fixture(scope="session")
def model_exllamav2(tmp_path_factory):
    return models.exllamav2(
        model_path="blockblockblock/TinyLlama-1.1B-Chat-v1.0-bpw4-exl2",
        device="cpu"
    )


@pytest.fixture(scope="session")
def model_mamba(tmp_path_factory):
    return models.mamba(
        model_name="state-spaces/mamba-130m-hf",
        device="cpu"
    )

ALL_MODEL_FIXTURES = ("model_llamacpp", "model_mlxlm", "model_transformers", "model_vllm", "model_exllamav2", "model_mamba")
"""


ALL_MODEL_FIXTURES = (
    "model_llamacpp",
    "model_mlxlm",
    "model_transformers",
    "model_vllm",
)


NOT_IMPLEMENTED = {
    "batch": ["model_llamacpp"],
    "stream": ["model_vllm"],
    "beam_search": ["model_llamacpp"],
}


def enforce_not_implemented(task_name, model_fixture):
    """
    Per `NOT_IMPLEMENTED`, mapping, if a model hasn't implemented a task,
    assert an NotImplementedError is raised. Otherwise, run normally
    """
    if model_fixture in NOT_IMPLEMENTED.get(task_name, []):
        return pytest.raises(NotImplementedError)
    else:
        return contextlib.nullcontext()


@pytest.mark.parametrize("sampler_name", ("greedy", "multinomial", "beam_search"))
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_text(request, model_fixture, sampler_name):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model, getattr(samplers, sampler_name)())
    with enforce_not_implemented(sampler_name, model_fixture):
        res = generator("test", max_tokens=10)
        assert isinstance(res, str)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_batch_text(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented("batch", model_fixture):
        res = generator(["test", "test2"], max_tokens=10)
        assert isinstance(res, list)
        assert isinstance(res[0], str)


@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_stream(request, model_fixture):
    model = request.getfixturevalue(model_fixture)
    generator = generate.text(model)
    with enforce_not_implemented("stream", model_fixture):
        for token in generator.stream("a b c ", max_tokens=10):
            assert isinstance(token, str)


@pytest.mark.parametrize(
    "pattern",
    (
        "[0-9]",
        "abc*",
        "\\+?[1-9][0-9]{7,14}",
    ),
)
@pytest.mark.parametrize("model_fixture", ALL_MODEL_FIXTURES)
def test_generate_regex(request, model_fixture, pattern):
    model = request.getfixturevalue(model_fixture)
    generator = generate.regex(model, pattern)
    res = generator("foobarbaz", max_tokens=20)
    assert re.match(pattern, res) is not None, res
