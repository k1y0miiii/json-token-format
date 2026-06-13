# jtf (Python)

Reference Python implementation of **JTF — JSON Token Format**, a lossless,
token-efficient encoding of the JSON data model for LLM prompts.

See the [project SPEC.md](../SPEC.md) for the authoritative format
specification and the [top-level README](../README.md) for an overview.

## Install

```sh
pip install -e .            # from this directory
# optional: real tokenizer for production dictionary decisions
pip install -e ".[tiktoken]"
```

## Use

```python
from jtf import encode, decode

text = encode({"id": 1, "name": "Ada"})   # JSON-compatible object -> JTF
obj  = decode(text)                        # JTF -> object
assert obj == {"id": 1, "name": "Ada"}     # round-trip is lossless
```

By default `encode` uses a deterministic heuristic cost function so output is
reproducible everywhere (this is what the conformance suite checks). To tune
the value dictionary to a real GPT tokenizer:

```python
from jtf import encode, tiktoken_cost
text = encode(data, cost_fn=tiktoken_cost)
```

## CLI

```sh
jtf encode input.json -o output.jtf
jtf decode output.jtf  -o roundtrip.json
jtf encode input.json --tiktoken     # use tiktoken for dictionary decisions
```

## Tests

```sh
pytest
```
