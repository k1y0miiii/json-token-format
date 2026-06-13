"""JTF — JSON Token Format.

A lossless, round-trippable encoding of the JSON data model that reduces
LLM tokenizer token count when passing structured data to language models.

Public API::

    from jtf import encode, decode

    text = encode(data)   # JSON-compatible Python object -> JTF string
    obj  = decode(text)   # JTF string -> Python object

Round-trip guarantee: ``decode(encode(x)) == x`` for any JSON-serializable
Python value.

See the project ``SPEC.md`` for the authoritative format specification.
"""

from .core import encode, decode
from .cost import heuristic_cost, tiktoken_cost

__all__ = ["encode", "decode", "heuristic_cost", "tiktoken_cost", "__version__"]

__version__ = "1.0.0"
