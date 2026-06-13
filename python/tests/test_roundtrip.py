"""Round-trip and behavioural tests for the jtf package.

The contract is ``decode(encode(x)) == x`` for every JSON-serializable ``x``.
"""

import json

import pytest

from jtf import encode, decode


# ---------------------------------------------------------------------------
# Round-trip cases — (name, value)
# ---------------------------------------------------------------------------

ROUNDTRIP_CASES = [
    # primitives
    ("null", None),
    ("true", True),
    ("false", False),
    ("int zero", 0),
    ("negative int", -42),
    ("float", 3.14),
    ("big int", 9007199254740993),
    ("negative float", -0.5),
    ("exp float", 1.5e10),
    # numeric-looking strings (must stay strings)
    ("num-str 007", "007"),
    ("num-str 3.14", "3.14"),
    ("num-str 1e5", "1e5"),
    ("num-str true", {"flag": "true"}),
    ("num-str false", {"flag": "false"}),
    ("num-str null", {"x": "null"}),
    ("num-str -1", {"-1": "value", "k": "-1"}),
    ("num-str 0.0", "0.0"),
    ("num-str 1E10", "1E10"),
    # empty values
    ("empty string", ""),
    ("empty object", {}),
    ("empty array", []),
    ("empty obj val", {"a": {}}),
    ("empty arr val", {"a": []}),
    ("empty nested", {"a": {"b": {}}}),
    ("empty string value", {"k": ""}),
    # special-character strings
    ("string with =", "key=value"),
    ("string with :", "key: value"),
    ("string with tab", "a\tb"),
    ("string with newline", "line1\nline2"),
    ("string with comma", "one, two, three"),
    ("string with quote", 'say "hello"'),
    ("string with backslash", "C:\\Users\\test"),
    ("string starting with #", "#hashtag"),
    ("string starting with $", "$money"),
    ("string starting with @", "@handle"),
    ("string with equals at end", {"k": "val="}),
    ("string with pipe", "a|b|c"),
    ("string with brackets", "arr[0]"),
    ("string with braces", "{obj}"),
    ("cyrillic", "Привет мир"),
    ("emoji", "test☃"),
    ("mixed cyrillic emoji", {"msg": "Привет \U0001F600"}),
    ("leading space string", {"k": " leading"}),
    ("trailing space string", {"k": "trailing "}),
    # objects
    ("flat object", {"name": "Иван", "age": 30, "active": True, "score": 9.5, "notes": None}),
    ("nested object", {"app": "test", "db": {"host": "localhost", "port": 5432, "pool": {"min": 2, "max": 10}}}),
    ("deeply nested 4 levels", {"l1": {"l2": {"l3": {"l4": {"value": "deep", "n": 42}}}}}),
    ("deeply nested 5 levels", {"a": {"b": {"c": {"d": {"e": "bottom"}}}}}),
    ("object reserved-word keys", {"null": 1, "true": 2, "false": 3}),
    ("object with = in key", {"ke=y": "val"}),
    ("object with : in key", {"ke:y": "val"}),
    ("object with tab in key", {"ke\ty": "val"}),
    ("object cyrillic key", {"имя": "Иван"}),
    # arrays
    ("primitive int array", [1, 2, 3, 4, 5]),
    ("primitive string array", ["alpha", "beta", "gamma"]),
    ("mixed primitive array", [1, "two", None, True, 3.14]),
    ("array of arrays", [[1, 2], [3, 4], [5, 6]]),
    ("array of arrays nested", [[1, [2, 3]], [4, [5, 6]]]),
    ("mixed/heterogeneous array", [{"a": 1}, {"a": 1, "b": 2}, "plain string", 42, None]),
    ("uniform flat array", [
        {"id": 1, "name": "Alice", "active": True},
        {"id": 2, "name": "Bob", "active": False},
        {"id": 3, "name": "Клод", "active": True},
    ]),
    ("uniform array with nulls", [
        {"x": 1, "y": None, "z": "a"},
        {"x": 2, "y": 99, "z": "b"},
    ]),
    ("uniform array cyrillic keys", [
        {"id": 1, "имя": "Алексей", "роль": "admin"},
        {"id": 2, "имя": "Мария", "роль": "editor"},
    ]),
    ("nested uniform array", [
        {"id": 1, "addr": {"city": "Moscow", "zip": "101000"}},
        {"id": 2, "addr": {"city": "Kazan", "zip": "420000"}},
        {"id": 3, "addr": {"city": "Samara", "zip": "443000"}},
    ]),
    ("deeply nested uniform array", [
        {"id": i, "meta": {"stats": {"score": float(i), "rank": i}}} for i in range(5)
    ]),
    ("array of empty objects", [{}, {}, {}]),
    ("single-element array", [42]),
    ("single-element obj array", [{"x": 1}]),
    ("large uniform table", [
        {"id": i, "val": f"item_{i}", "score": float(i) * 1.5, "ok": i % 2 == 0}
        for i in range(20)
    ]),
    ("key with space tabular", [
        {"first name": "Ann", "age": 1},
        {"first name": "Bob", "age": 2},
    ]),
    # nested containers in objects
    ("nested array in object", {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], "count": 2}),
    ("array of arrays in object", {"matrix": [[1, 0], [0, 1]]}),
    ("url in value", {"url": "https://example.com/path?q=1&r=2", "title": "Example"}),
    ("timestamps", {"created": "2024-11-18T14:32:07Z", "updated": "2024-11-19T00:00:00Z"}),
    # value-dictionary boundary cases
    ("dict boundary $1 vs $10", {
        **{f"v{i}": chr(97 + i) * 4 + "_" + chr(97 + i) * 4 + "_" + chr(97 + i) * 4 + "_" + chr(97 + i) * 4 for i in range(11)},
        **{f"r{i}": chr(97 + i) * 4 + "_" + chr(97 + i) * 4 + "_" + chr(97 + i) * 4 + "_" + chr(97 + i) * 4 for i in range(11)},
    }),
    ("dict value with = sign", [
        {"status": "key=value", "other": "key=value"},
        {"status": "key=value", "other": "different"},
    ]),
    ("dict value that looks numeric", [
        {"v": "3.14", "other": "3.14"},
        {"v": "3.14", "other": "something"},
    ]),
    ("no dict when not profitable", {"x": "ab", "y": "ab"}),
    # top-level primitives
    ("top-level string", "hello world"),
    ("top-level int", 42),
    ("top-level null", None),
    ("top-level bool", True),
    ("top-level float", 3.14),
]


@pytest.mark.parametrize("name,value", ROUNDTRIP_CASES, ids=[c[0] for c in ROUNDTRIP_CASES])
def test_roundtrip(name, value):
    encoded = encode(value)
    assert isinstance(encoded, str)
    decoded = decode(encoded)
    assert decoded == value, f"{name}: {value!r} -> {encoded!r} -> {decoded!r}"


# ---------------------------------------------------------------------------
# Specific output-shape assertions (the format, not just round-trip)
# ---------------------------------------------------------------------------

def test_kv_separator_is_equals():
    assert encode({"a": 1}) == "a=1"


def test_nested_block_uses_colon_and_tab():
    out = encode({"a": {"b": 1}})
    assert out == "a:\n\tb=1"


def test_primitive_array_inline():
    assert encode([1, 2, 3]) == "[3] 1,2,3"


def test_empty_containers():
    assert encode({}) == "{}"
    assert encode([]) == "[0]"
    assert encode({"a": {}}) == "a={}"
    assert encode({"a": []}) == "a=[0]"


def test_tabular_header_flat():
    out = encode([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    lines = out.split("\n")
    assert lines[0] == "#2 id name"
    assert lines[1] == "\t1\tA"
    assert lines[2] == "\t2\tB"


def test_tabular_header_nested_dotted():
    out = encode([
        {"id": 1, "addr": {"city": "Moscow"}},
        {"id": 2, "addr": {"city": "Kazan"}},
    ])
    assert out.split("\n")[0] == "#2 id addr.city"


def test_numeric_string_quoted():
    assert encode("007") == '"007"'
    assert encode("3.14") == '"3.14"'


def test_reserved_word_string_quoted():
    assert encode("true") == '"true"'
    assert encode("null") == '"null"'


def test_dictionary_emitted_for_repeated_values():
    data = [{"s": "a_repeated_long_value_here"} for _ in range(10)]
    out = encode(data)
    assert out.startswith("#vdf:v2\n#dict:\n")
    assert "#end" in out
    assert decode(out) == data


def test_no_dictionary_when_unprofitable():
    out = encode({"x": "ab", "y": "ab"})
    assert "#vdf" not in out


def test_decode_is_inverse_of_sample_files(tmp_path):
    # exercise a realistic mixed document
    data = {
        "meta": {"id": "req_1", "ok": True},
        "rows": [
            {"k": "active", "n": 1},
            {"k": "active", "n": 2},
            {"k": "pending", "n": 3},
        ],
        "tags": ["a", "b", "c"],
    }
    assert decode(encode(data)) == data
