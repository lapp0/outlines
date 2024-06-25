from typing import List

import torch
from torch.nn import functional as F

from .base_logits_processor import OutlinesLogitsProcessor


class TemperatureLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply temperature scaling to logits.

    Args:
        temperature (float): The temperature value to scale logits. Must be > 0.
                            A value of 0 will result in greedy sampling.

    Raises:
        ValueError: If temperature is less than 0.
    """

    def __init__(self, temperature=0):
        if temperature < 0:
            raise ValueError("Temperature must be > 0.")
        self.temperature = temperature

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        For temperature 0, set the highest logit to inf (equivalent to greedy sampling)
        """
        if self.temperature == 0:
            max_indices = logits.argmax(dim=-1, keepdim=True)
            mask = torch.full_like(logits, -torch.inf).scatter_(-1, max_indices, 0)
            return logits + mask
        return logits / self.temperature


class MinPLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to ensure a minimum probability for each element in the logits.

    Args:
        min_p (float): The minimum probability value. Must be between 0 and 1.

    Raises:
        ValueError: If min_p is not between 0 and 1.
    """

    def __init__(self, min_p=0.01):
        if min_p <= 0 or min_p >= 1:
            raise ValueError("min_p must be between 0 and 1")
        self.min_p = min_p

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure each logit's probability is at least min_p by
        setting logits with lower probabilities to -inf.
        """
        return logits.masked_fill(F.softmax(logits, dim=-1) < self.min_p, -torch.inf)


class TopPLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply top-p (nucleus) filtering to logits.

    Args:
        p (float): The cumulative probability threshold. Must be between 0 and 1.

    Raises:
        ValueError: If p is not between 0 and 1.
    """

    def __init__(self, p: float = 0.9):
        if p <= 0 or p > 1:
            raise ValueError("p must be between 0 and 1")
        self.p = p

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply top-p filtering by setting lowest logits
        with cumulative probability above p to -inf.
        """
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumulative_probs > self.p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = 0
        return logits.masked_fill(mask.scatter(1, sorted_indices, mask), -float("inf"))


class TopKLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply top-k filtering to logits.

    Args:
        k (int): The number of highest probability tokens to keep.

    Raises:
        ValueError: If k is not a positive integer.
    """

    def __init__(self, k: int = 50):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply top-k filtering by keeping the top k logits and setting the rest to -inf.
        """
        if self.k > logits.size(-1):
            raise ValueError("k cannot be greater than the number of logits")
        top_k = torch.topk(logits, self.k, dim=-1).indices
        mask = torch.ones_like(logits, dtype=torch.bool).scatter_(-1, top_k, False)
        return logits.masked_fill(mask, -torch.inf)


class TFSLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply Tail Free Sampling (TFS) to logits."""

    def __init__(self, threshold: float = 0.9):
        if threshold <= 0 or threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply TFS by keeping top logits with cumulative probability
        below the threshold and setting others to -inf.
        """
        if logits.numel() == 0:
            return logits
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_keep = cumulative_probs <= self.threshold

        last_valid_index = sorted_indices_to_keep.sum(dim=-1, keepdim=True) - 1
        sorted_indices_to_keep.scatter_(1, last_valid_index, 1)

        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(
            1, sorted_indices, sorted_indices_to_keep
        )
        return logits.masked_fill(~mask, -torch.inf)


class QuadraticSmoothingLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply quadratic smoothing to logits."""

    def __init__(self, alpha: float = 0.5):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply quadratic smoothing by mixing logits with their square, controlled by alpha.
        """
        return logits * (1 - self.alpha) + logits.pow(2) * self.alpha


class RepetitionPenaltyLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply a repetition penalty to logits.

    Args:
        penalty (float): The penalty to apply to repeated tokens. Must be > 0.
    """

    def __init__(self, penalty: float = 1.2):
        if penalty < 0:
            raise ValueError("penalty must be <= 1")
        self.penalty = penalty

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a penalty to logits of tokens that appear
        in the input sequence to reduce repetition.
        """
        input_ids_tensor = torch.tensor(input_ids, device=logits.device)
        token_counts = torch.bincount(
            input_ids_tensor.view(-1), minlength=logits.size(-1)
        ).float()
        return logits / (1 + token_counts * (self.penalty - 1)).unsqueeze(0)


class PresencePenaltyLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply a presence penalty to logits.

    Args:
        penalty (float): The penalty to apply to tokens present in the sequence. Must be > 0.
    """

    def __init__(self, penalty: float = 0.1):
        if penalty < 0:
            raise ValueError("penalty must be greater than or equal to 0")
        self.penalty = penalty

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a contsant penalty to logits of tokens that are present in the input sequence.
        """
        input_ids_tensor = torch.tensor(input_ids, device=logits.device)
        unique_tokens = torch.unique(input_ids_tensor, sorted=False)
        logits[:, unique_tokens] -= self.penalty
        return logits


class FrequencyPenaltyLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to apply a frequency penalty to logits based on token frequency.

    Args:
        penalty (float): The penalty to apply to tokens based on frequency. Must be > 0.
    """

    def __init__(self, penalty: float = 0.1):
        if penalty < 0:
            raise ValueError("penalty must be greater than or equal to 0")
        self.penalty = penalty

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply a penalty to logits based on a multiple of the frequenccy of tokens in the input sequence.
        """
        input_ids_tensor = torch.tensor(input_ids, device=logits.device)
        token_counts = torch.bincount(
            input_ids_tensor.view(-1), minlength=logits.size(-1)
        ).float()
        return logits - token_counts * self.penalty


class NoRepeatNGramLogitsProcessor(OutlinesLogitsProcessor):
    """Processor to ensure no repeated n-grams in the sequence.

    Args:
        n (int): The n-gram size to check for repetitions.
    """

    def __init__(self, n: int = 3):
        if n <= 0:
            raise ValueError("n must be greater than 0")
        self.n = n

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Set logits which would result in n-gram repetition exceeding `n` to -inf
        """
        if len(input_ids[0]) < self.n:
            return logits
        input_ids_tensor = torch.tensor(input_ids, device=logits.device)
        ngrams = [
            tuple(input_ids_tensor[0, i : i + self.n])
            for i in range(len(input_ids_tensor[0]) - self.n + 1)
        ]
        last_ngram = tuple(input_ids_tensor[0, -self.n + 1 :])
        mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
        # TODO: don't iterate
        for token in range(logits.shape[-1]):
            if tuple(list(last_ngram) + [token]) in ngrams:
                mask[token] = True
        logits.masked_fill_(mask, -float("inf"))
        return logits
