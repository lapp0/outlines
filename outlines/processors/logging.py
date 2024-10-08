import logging
import sys
import warnings
from typing import List

import torch

from .base_logits_processor import OutlinesLogitsProcessor


class LogitsLoggingLogitsProcessor(OutlinesLogitsProcessor):
    """Handle chaining two logits processors to process logits sequentially"""

    def __init__(self, tokenizer, top_n=8, logger=None, warn=True):
        self.tokenizer = tokenizer
        self.top_n = top_n
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("logits_logger")
            self.logger.setLevel(logging.info)
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
        if warn:
            warnings.warn(
                "Do not use LoggingLogitsProcessor in production, it slows down generation."
            )

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        # all token probs for the current batch
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # top candidate tokens probs
        top_indices = torch.topk(probs, self.top_n).indices
        top_indices = [
            set(row.tolist()) | {self.tokenizer.eos_token_id} for row in top_indices
        ]
        batch_top_probs = [
            {token_idx: probs[batch_num, token_idx] for token_idx in row_indices}
            for batch_num, row_indices in enumerate(top_indices)
        ]
        self.logger.info(batch_top_probs)
        return logits


class SequenceLoggingLogitsProcessor(OutlinesLogitsProcessor):
    def __init__(self, tokenizer, top_n=8, logger=None, warn=True):
        self.tokenizer = tokenizer
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("sequence_logger")
            self.logger.setLevel(logging.info)
            self.logger.addHandler(logging.StreamHandler(sys.stderr))
        if warn:
            warnings.warn(
                "Do not use SequenceLoggingProcessor in production, it slows down generation."
            )

    def process_logits(
        self, input_ids: List[List[int]], logits: torch.Tensor
    ) -> torch.Tensor:
        self.logger.info(self.tokenizer.decode(input_ids))
        return logits
