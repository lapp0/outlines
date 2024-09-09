import pytest
import torch

from outlines.processors import (
    FrequencyPenaltyLogitsProcessor,
    MinPLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PresencePenaltyLogitsProcessor,
    QuadraticSmoothingLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsProcessor,
    TFSLogitsProcessor,
    TopKLogitsProcessor,
    TopPLogitsProcessor,
)


@pytest.fixture()
def input_logits():
    return torch.tensor([[1.0, 2.0, 3.0]])


@pytest.fixture()
def input_logits_2d():
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# TemperatureLogitsProcessor tests
def test_normal_temperature(input_logits):
    processor = TemperatureLogitsProcessor(temperature=0.5)
    expected = torch.tensor([[2.0, 4.0, 6.0]])
    torch.testing.assert_close(processor.process_logits([], input_logits), expected)


def test_temperature_one(input_logits):
    processor = TemperatureLogitsProcessor(temperature=1.0)
    expected = input_logits.clone()
    torch.testing.assert_close(processor.process_logits([], input_logits), expected)


def test_extreme_temperature_high(input_logits):
    processor = TemperatureLogitsProcessor(temperature=100.0)
    expected = input_logits / 100.0
    torch.testing.assert_close(processor.process_logits([], input_logits), expected)


def test_extreme_temperature_low(input_logits):
    processor = TemperatureLogitsProcessor(temperature=0.01)
    expected = input_logits / 0.01
    torch.testing.assert_close(processor.process_logits([], input_logits), expected)


def test_temperature_zero(input_logits):
    processor = TemperatureLogitsProcessor(temperature=0)
    result = processor.process_logits([], input_logits)
    max_idx = input_logits.argmax(dim=-1, keepdim=True).item()
    assert torch.softmax(result, dim=-1)[0][max_idx] == 1.0


def test_temperature_negative(input_logits):
    with pytest.raises(ValueError):
        TemperatureLogitsProcessor(temperature=-1.0)


# MinPLogitsProcessor tests
def test_valid_min_p(input_logits):
    processor = MinPLogitsProcessor(min_p=0.1)
    processed_logits = processor.process_logits([], input_logits)
    probs = torch.softmax(processed_logits, dim=-1)
    assert not torch.isinf(probs).any()
    assert (probs >= 0.1).any()  # Ensure at least one probability meets the minimum


def test_all_below_min_p(input_logits):
    processor = MinPLogitsProcessor(min_p=0.99)
    processed_logits = processor.process_logits([], input_logits)
    assert torch.isinf(processed_logits).all()


def test_some_below_min_p(input_logits):
    processor = MinPLogitsProcessor(min_p=0.5)
    probs = torch.softmax(input_logits, dim=-1)
    mask = probs < 0.5
    expected = input_logits.masked_fill(mask, -torch.inf)
    torch.testing.assert_close(processor.process_logits([], input_logits), expected)


def test_no_below_min_p(input_logits):
    processor = MinPLogitsProcessor(min_p=0.01)
    processed_logits = processor.process_logits([], input_logits)
    torch.testing.assert_close(processed_logits, input_logits)


def test_min_p_boundary_values():
    with pytest.raises(ValueError):
        MinPLogitsProcessor(min_p=0)

    with pytest.raises(ValueError):
        MinPLogitsProcessor(min_p=1)


def test_high_dimensional_logits():
    processor = MinPLogitsProcessor(min_p=0.1)
    input_logits = torch.randn(3, 4, 5)
    processed_logits = processor.process_logits([], input_logits)
    assert processed_logits.shape == input_logits.shape


# TopPLogitsProcessor tests
def test_valid_top_p(input_logits):
    processor = TopPLogitsProcessor(p=0.9)
    processed_logits = processor.process_logits([], input_logits)
    probs = torch.softmax(processed_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    assert (cumulative_probs <= 0.9).any()


def test_top_p_zero(input_logits):
    with pytest.raises(ValueError):
        TopPLogitsProcessor(p=0)


def test_top_p_one(input_logits):
    processor = TopPLogitsProcessor(p=1.0)
    torch.testing.assert_close(processor.process_logits([], input_logits), input_logits)


def test_top_p_greater_than_one(input_logits):
    with pytest.raises(ValueError):
        TopPLogitsProcessor(p=1.1)


# TopKLogitsProcessor tests
def test_valid_top_k(input_logits):
    processor = TopKLogitsProcessor(k=2)
    processed_logits = processor.process_logits([], input_logits)
    assert (processed_logits == float("-inf")).sum() == 1


def test_top_k_zero(input_logits):
    with pytest.raises(ValueError):
        TopKLogitsProcessor(k=0)


def test_top_k_equal_to_logits(input_logits):
    processor = TopKLogitsProcessor(k=3)
    torch.testing.assert_close(processor.process_logits([], input_logits), input_logits)


def test_top_k_greater_than_logits(input_logits):
    processor = TopKLogitsProcessor(k=4)
    with pytest.raises(ValueError):
        processor.process_logits([], input_logits)


# TFSLogitsProcessor tests
def test_valid_tfs(input_logits):
    processor = TFSLogitsProcessor(threshold=0.9)
    processed_logits = processor.process_logits([], input_logits)
    probs = torch.softmax(processed_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    assert (cumulative_probs <= 0.9).all() or (cumulative_probs <= 1).all()


def test_tfs_zero(input_logits):
    with pytest.raises(ValueError):
        TFSLogitsProcessor(threshold=0)


def test_tfs_one(input_logits):
    processor = TFSLogitsProcessor(threshold=1.0)
    torch.testing.assert_close(processor.process_logits([], input_logits), input_logits)


def test_tfs_greater_than_one(input_logits):
    with pytest.raises(ValueError):
        TFSLogitsProcessor(threshold=1.1)


# QuadraticSmoothingLogitsProcessor tests
def test_valid_quadratic_smoothing(input_logits):
    processor = QuadraticSmoothingLogitsProcessor(alpha=0.5)
    processed_logits = processor.process_logits([], input_logits)
    expected = input_logits**2 * 0.5 + input_logits * 0.5
    torch.testing.assert_close(processed_logits, expected)


def test_quadratic_smoothing_zero(input_logits):
    processor = QuadraticSmoothingLogitsProcessor(alpha=0.0)
    torch.testing.assert_close(processor.process_logits([], input_logits), input_logits)


def test_quadratic_smoothing_negative(input_logits):
    with pytest.raises(ValueError):
        QuadraticSmoothingLogitsProcessor(alpha=-0.1)


def test_quadratic_smoothing_greater_than_one(input_logits):
    with pytest.raises(ValueError):
        QuadraticSmoothingLogitsProcessor(alpha=1.1)


# RepetitionPenaltyLogitsProcessor tests
def test_valid_repetition_penalty(input_logits):
    processor = RepetitionPenaltyLogitsProcessor(penalty=1.5)
    input_ids = [[0, 1, 2]]
    processed_logits = processor.process_logits(input_ids, torch.tensor(input_logits))
    expected = input_logits.clone()
    for seq in input_ids:
        for token in seq:
            expected[..., token] /= 1.5
    torch.testing.assert_close(processed_logits, expected)


def test_repetition_penalty_one(input_logits):
    processor = RepetitionPenaltyLogitsProcessor(penalty=1.0)
    processed_logits = processor.process_logits([[0, 1, 2]], input_logits)
    torch.testing.assert_close(processed_logits, input_logits)


def test_repetition_penalty_negative(input_logits):
    with pytest.raises(ValueError):
        RepetitionPenaltyLogitsProcessor(penalty=-0.5)


# PresencePenaltyLogitsProcessor tests
def test_valid_presence_penalty(input_logits):
    processor = PresencePenaltyLogitsProcessor(penalty=0.5)
    input_ids = [[0, 1, 2]]
    processed_logits = processor.process_logits(input_ids, torch.tensor(input_logits))
    expected = torch.tensor(input_logits)
    expected[0] -= 0.5
    torch.testing.assert_close(processed_logits, expected)


def test_presence_penalty_zero(input_logits):
    processor = PresencePenaltyLogitsProcessor(penalty=0.0)
    torch.testing.assert_close(
        processor.process_logits([[0, 1, 2]], input_logits), input_logits
    )


def test_presence_penalty_negative(input_logits):
    with pytest.raises(ValueError):
        PresencePenaltyLogitsProcessor(penalty=-0.5)


# FrequencyPenaltyLogitsProcessor tests
def test_valid_frequency_penalty(input_logits):
    processor = FrequencyPenaltyLogitsProcessor(penalty=0.5)
    input_ids = [[0, 1, 2]]
    processed_logits = processor.process_logits(input_ids, torch.tensor(input_logits))
    expected = input_logits.clone()
    token_counts = torch.zeros(
        input_logits.shape[-1], dtype=input_logits.dtype, device=input_logits.device
    )
    for seq in input_ids:
        for token in seq:
            token_counts[token] += 1
    for token in range(input_logits.shape[-1]):
        expected[..., token] -= token_counts[token] * 0.5
    torch.testing.assert_close(processed_logits, expected)


def test_frequency_penalty_zero(input_logits):
    processor = FrequencyPenaltyLogitsProcessor(penalty=0.0)
    torch.testing.assert_close(
        processor.process_logits([[0, 1, 2]], input_logits), input_logits
    )


def test_frequency_penalty_negative(input_logits):
    with pytest.raises(ValueError):
        FrequencyPenaltyLogitsProcessor(penalty=-0.5)


# NoRepeatNGramLogitsProcessor tests
def test_valid_no_repeat_ngram(input_logits):
    processor = NoRepeatNGramLogitsProcessor(n=2)
    processed_logits = processor.process_logits(
        [[0, 1, 0, 1]], torch.tensor(input_logits)
    )
    assert processed_logits[0, 0] == float("-inf")


def test_no_repeat_ngram_zero(input_logits):
    with pytest.raises(ValueError):
        NoRepeatNGramLogitsProcessor(n=0)


def test_no_repeat_ngram_one(input_logits):
    processor = NoRepeatNGramLogitsProcessor(n=1)
    torch.testing.assert_close(
        processor.process_logits([[0, 1, 2]], input_logits), input_logits
    )


def test_no_repeat_ngram_greater_than_sequence_length(input_logits):
    processor = NoRepeatNGramLogitsProcessor(n=10)
    torch.testing.assert_close(
        processor.process_logits([[0, 1, 2]], input_logits), input_logits
    )
