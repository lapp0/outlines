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
    FSMLogitsProcessor,
    JSONLogitsProcessor,
    RegexLogitsProcessor,
)

# aliases for convenience
chained = ChainedLogitsProcessor

cfg = CFGLogitsProcessor
fsm = FSMLogitsProcessor
json = JSONLogitsProcessor
regex = RegexLogitsProcessor

temperature = TemperatureLogitsProcessor
min_p = MinPLogitsProcessor

sequence_logging = SequenceLoggingLogitsProcessor
logits_logging = LogitsLoggingLogitsProcessor
