#!/usr/bin/env python3
"""Generate the conformance input JSON files and expected JTF outputs.

The Python encoder is the source of truth. This script writes, for each case:
  cases/NNN-name.json   the input (canonical JSON)
  cases/NNN-name.jtf    encode(input) under the canonical heuristic cost fn
and a manifest.json listing every case.

The expected .jtf files are the *contract*: both the Python and JS libraries
must reproduce them exactly, and decode(.jtf) must equal the .json input.

Run from anywhere:  python3 conformance/_gen_inputs.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# import the installed/in-tree jtf package
sys.path.insert(0, os.path.join(HERE, "..", "python", "src"))
from jtf import encode  # noqa: E402

CASES_DIR = os.path.join(HERE, "cases")


# Ordered (name, value). Numbering is assigned by position.
CASES = [
    # ---- scalars ----
    ("scalar-null", None),
    ("scalar-true", True),
    ("scalar-false", False),
    ("scalar-int", 42),
    ("scalar-int-zero", 0),
    ("scalar-int-negative", -42),
    ("scalar-float", 3.14),
    ("scalar-float-negative", -0.5),
    # NOTE: integral floats (e.g. 1000000.0) and float exponents (1.5e10) are
    # intentionally excluded from the shared vectors: JavaScript's single Number
    # type collapses them to integers at JSON.parse time (1000000.0 -> 1000000),
    # so the ".0" cannot be reproduced cross-language. This is a property of the
    # JSON data model in JS, not a JTF defect. Non-integral floats round-trip
    # identically in both languages and are covered below.
    ("scalar-float-small", 0.0009765625),
    ("scalar-string", "hello world"),
    ("scalar-empty-string", ""),
    # ---- numbers / floats edge cases ----
    ("numbers-mixed-array", [1, 2, 3, -4, 0, 100000]),
    ("numbers-floats", {"a": 0.1, "b": 2.5, "c": -7.25, "d": 123.456}),
    ("numbers-numeric-strings", {"zip": "007", "pi": "3.14", "exp": "1e5", "neg": "-1"}),
    # ---- null / empty ----
    ("empty-object", {}),
    ("empty-array", []),
    ("empty-object-value", {"a": {}}),
    ("empty-array-value", {"a": []}),
    ("empty-nested", {"a": {"b": {}}}),
    ("nulls-in-object", {"x": None, "y": 1, "z": None}),
    # ---- scalars in a flat object ----
    ("object-flat", {"name": "Ada", "age": 30, "active": True, "score": 9.5, "notes": None}),
    ("object-reserved-keys", {"null": 1, "true": 2, "false": 3}),
    # ---- nested objects ----
    ("object-nested", {"app": "svc", "db": {"host": "localhost", "port": 5432, "pool": {"min": 2, "max": 10}}}),
    ("object-deep-5-levels", {"a": {"b": {"c": {"d": {"e": "bottom"}}}}}),
    # ---- uniform tabular arrays (flat) ----
    ("tabular-flat", [
        {"id": 1, "name": "Alice", "active": True},
        {"id": 2, "name": "Bob", "active": False},
        {"id": 3, "name": "Carol", "active": True},
    ]),
    ("tabular-flat-with-nulls", [
        {"x": 1, "y": None, "z": "a"},
        {"x": 2, "y": 99, "z": "b"},
    ]),
    ("tabular-key-with-space", [
        {"first name": "Ann", "age": 1},
        {"first name": "Bob", "age": 2},
    ]),
    # ---- uniform tabular arrays (nested -> dotted paths) ----
    ("tabular-nested-dotted", [
        {"id": 1, "addr": {"city": "Moscow", "zip": "101000"}},
        {"id": 2, "addr": {"city": "Kazan", "zip": "420000"}},
        {"id": 3, "addr": {"city": "Samara", "zip": "443000"}},
    ]),
    # ---- non-uniform / mixed arrays ----
    ("array-primitive-ints", [1, 2, 3, 4, 5]),
    ("array-primitive-strings", ["alpha", "beta", "gamma"]),
    ("array-mixed-primitives", [1, "two", None, True, 3.14]),
    ("array-of-arrays", [[1, 2], [3, 4], [5, 6]]),
    ("array-heterogeneous", [{"a": 1}, {"a": 1, "b": 2}, "plain", 42, None]),
    ("array-empty-objects", [{}, {}, {}]),
    # ---- unicode ----
    ("unicode-cyrillic", {"msg": "Привет мир", "имя": "Иван"}),
    ("unicode-emoji", {"a": "test☃", "b": "Привет 😀"}),
    ("unicode-tabular", [
        {"id": 1, "имя": "Алексей", "роль": "admin"},
        {"id": 2, "имя": "Мария", "роль": "editor"},
    ]),
    # ---- strings needing escaping ----
    ("escape-special-chars", {
        "eq": "key=value",
        "colon": "key: value",
        "tab": "a\tb",
        "newline": "line1\nline2",
        "comma": "one, two",
        "quote": 'say "hi"',
        "backslash": "C:\\Users\\x",
        "hash": "#tag",
        "dollar": "$money",
        "at": "@handle",
        "pipe": "a|b",
    }),
    ("escape-keys", {"ke=y": 1, "ke:y": 2, "ke\ty": 3}),
    # ---- value dictionary (repeated values, profitable) ----
    ("dict-repeated-values", [
        {"id": 1, "status": "active", "plan": "premium", "region": "eu-west-1", "tier": "gold"},
        {"id": 2, "status": "pending", "plan": "basic", "region": "eu-west-1", "tier": "silver"},
        {"id": 3, "status": "active", "plan": "premium", "region": "eu-west-1", "tier": "gold"},
        {"id": 4, "status": "suspended", "plan": "basic", "region": "us-east-1", "tier": "bronze"},
        {"id": 5, "status": "active", "plan": "premium", "region": "eu-west-1", "tier": "gold"},
        {"id": 6, "status": "pending", "plan": "basic", "region": "ap-south-1", "tier": "silver"},
        {"id": 7, "status": "active", "plan": "premium", "region": "eu-west-1", "tier": "gold"},
        {"id": 8, "status": "suspended", "plan": "basic", "region": "us-east-1", "tier": "bronze"},
    ]),
    ("dict-url-prefix", [
        {"id": 1, "url": "https://api.example.com/v2/users/1"},
        {"id": 2, "url": "https://api.example.com/v2/users/2"},
        {"id": 3, "url": "https://api.example.com/v2/users/3"},
        {"id": 4, "url": "https://api.example.com/v2/users/4"},
    ]),
    # 11 distinct long values, each repeated -> forces $0..$10 dict entries and
    # exercises the $1-vs-$10 disambiguation in the body.
    ("dict-exact-values", [
        {f"k{j}": chr(97 + j) * 40 for j in range(11)} for _ in range(2)
    ]),
    ("dict-not-profitable", {"x": "ab", "y": "ab"}),
    # ---- deep nesting / combined ----
    ("nested-array-in-object", {"users": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], "count": 2}),
    ("combined-realistic", {
        "meta": {"requestId": "req_8f2a", "tookMs": 42, "apiVersion": "2026-04-01"},
        "page": {"n": 1, "size": 12, "total": 87},
        "data": [
            {"id": 100001, "title": "Add dark mode", "state": "open", "comments": 2},
            {"id": 100002, "title": "Fix memory leak", "state": "closed", "comments": 7},
            {"id": 100003, "title": "Improve docs", "state": "open", "comments": 0},
        ],
        "tags": ["bug", "docs", "question"],
    }),
]


def main():
    os.makedirs(CASES_DIR, exist_ok=True)
    manifest = []
    for i, (name, value) in enumerate(CASES, start=1):
        num = f"{i:03d}"
        stem = f"{num}-{name}"
        json_path = os.path.join(CASES_DIR, stem + ".json")
        jtf_path = os.path.join(CASES_DIR, stem + ".jtf")

        # canonical JSON input (stable, ensure_ascii=False, 2-space indent)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2))
            f.write("\n")

        encoded = encode(value)
        with open(jtf_path, "w", encoding="utf-8") as f:
            f.write(encoded)
            f.write("\n")

        manifest.append({"id": num, "name": name, "input": stem + ".json", "expected": stem + ".jtf"})

    with open(os.path.join(HERE, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"version": "1.0", "count": len(manifest), "cases": manifest},
                           ensure_ascii=False, indent=2))
        f.write("\n")

    print(f"Generated {len(manifest)} conformance cases in {CASES_DIR}")


if __name__ == "__main__":
    main()
