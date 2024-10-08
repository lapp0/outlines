from .base_logits_processor import ChainedLogitsProcessor, OutlinesLogitsProcessor
from .logging import LogitsLoggingLogitsProcessor, SequenceLoggingLogitsProcessor
from .sampling import (
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
from .structured import (
    CFGLogitsProcessor,
    GuideLogitsProcessor,
    JSONLogitsProcessor,
    RegexLogitsProcessor,
)

# aliases for convenience
chained = ChainedLogitsProcessor

cfg = CFGLogitsProcessor
guide = GuideLogitsProcessor
json = JSONLogitsProcessor
regex = RegexLogitsProcessor

temperature = TemperatureLogitsProcessor
min_p = MinPLogitsProcessor

sequence_logging = SequenceLoggingLogitsProcessor
logits_logging = LogitsLoggingLogitsProcessor
