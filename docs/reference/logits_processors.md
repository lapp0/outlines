# Logits Processors

TODO: Explanation of what logits processors do

TODO: List of `OutlinesLogitsProcessor`

TODO: Example of how to implement TemperatureLogitsProcessor

TODO: using logits processors with models directly vs using with outlines

TODO: Explanation of pipelines

TODO: Link to log logits

## Using Logits Processors in Outlines

TODO Explanation

```
import outlines
```


## Chaining Logits Processors

```
import outlines
import outlines.processors as processors

model = outlines.models.llamacpp(
    repo_id="M4-ai/TinyMistral-248M-v2-Instruct-GGUF",
    filename="TinyMistral-248M-v2-Instruct.Q4_K_M.gguf"
)

# Create a chained logits processor
logits_processor = (
    processors.sequence_logging(model.tokenizer) |  # Log the generated sequence
	processors.logits_logging(model.tokenizer) |  # Log the raw logits
	processors.regex(r"[0-9]*", model.tokenizer) |  # Restrict the logits to match the pattern
	processors.temperature(0.5) |  # Set temperature to 0.5
	processors.logits_logging(model.tokenizer)  # Log the restricted, temperature-augmentent, sampled logits
)

generator = outlines.generate.base(model, logits_process)
generator("What is your favorite number? ")
```

Output:
```
TODO
```
