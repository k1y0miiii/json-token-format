"""Token cost functions for JTF's value-dictionary break-even analysis.

The value dictionary is only emitted when it is *profitable*, which requires
estimating how many tokens a given string costs. JTF supports two cost
functions:

* :func:`heuristic_cost` — a deterministic, dependency-free estimate of
  ``max(1, len(s) // 4)`` tokens. This is the **canonical** cost function used
  by the conformance suite, because it produces byte-identical output across
  languages and environments without requiring a tokenizer to be installed.

* :func:`tiktoken_cost` — uses OpenAI's ``tiktoken`` (``cl100k_base``) when
  available, giving the most accurate real-world token savings. This is what
  you want in production when targeting GPT-family models. Output may differ
  from the heuristic on borderline dictionary decisions.

``encode(data)`` uses :func:`heuristic_cost` by default so that results are
reproducible. Pass ``cost_fn=tiktoken_cost`` to optimize for a real tokenizer.
"""

from __future__ import annotations

from typing import Callable

CostFn = Callable[[str], int]


def heuristic_cost(s: str) -> int:
    """Deterministic token-count estimate: ``max(1, len(s) // 4)``.

    This is the canonical cost function for the JTF conformance suite. It
    depends on nothing and yields identical results in every language, which
    is what makes the golden vectors reproducible.
    """
    return max(1, len(s) // 4)


_TIKTOKEN_ENC = None
_TIKTOKEN_TRIED = False


def tiktoken_cost(s: str) -> int:
    """Token count via ``tiktoken`` (``cl100k_base``); falls back to the
    heuristic if ``tiktoken`` is not installed.

    Use this when you want JTF's dictionary decisions tuned to a real
    GPT-family tokenizer. Note that output may differ from the canonical
    heuristic-based encoding on borderline cases.
    """
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if _TIKTOKEN_ENC is None and not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:  # pragma: no cover - environment dependent
            import tiktoken

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # pragma: no cover
            _TIKTOKEN_ENC = None
    if _TIKTOKEN_ENC is None:  # pragma: no cover
        return heuristic_cost(s)
    return len(_TIKTOKEN_ENC.encode(s))
