# jtf — JSON Token Format v2

A lossless, round-trippable encoding of the JSON data model that reduces tiktoken
token count when passing structured data to LLMs.

## Format spec

### Primitives

`null`, `true`, `false` written literally. Numbers in canonical JSON form (no
trailing zeros). Strings are unquoted when safe; double-quoted with standard JSON
escape sequences otherwise.

A string is safe to leave unquoted when it:
- is non-empty and not `null`/`true`/`false`
- has no leading/trailing whitespace
- contains none of `= : | " \ # [ ] { } $ @ tab newline`
- would not be mis-parsed as a number or boolean on decode

Numeric-looking strings (`"007"`, `"3.14"`, `"1e5"`) are always double-quoted to
prevent lossy round-trips.

### Objects (tab-indented)

```
key=value          <- primitive value (= separator, 1 token)
key:               <- nested block follows on next indented lines
    child=...
```

Nesting uses one tab per level. Empty object: `{}`.

### Arrays

**Empty:** `[0]`

**Primitive (inline, comma-separated):**
```
[N] v1,v2,v3
```

**Tabular — flat-uniform objects (all items share same keys, all-primitive values):**
```
#N k1 k2 k3
    v1    v2    v3
    v4    v5    v6
```
Header is `#N` followed by space-separated key names. Rows are tab-indented, cells
are tab-separated. Keys with spaces or special characters use a `csv:` prefix:
```
#N csv:"key one","key two"
    cell1    cell2
```

**Tabular — nested-uniform objects (uniform with nested dicts, all leaf values
primitive):** same `#N` header format, but keys use dotted paths:
```
#N id addr.city addr.zip
    1    Moscow    101000
    2    Kazan     420000
```
On decode, dotted paths reconstruct the nested dict structure.

**Mixed/heterogeneous (dash list):**
```
[N]
    - 42
    - some string
    - k=v
```

### Value dictionary (optional)

When the same string values appear repeatedly and substitution saves tokens (net of
the header overhead), a dictionary block is prepended:

```
#vdf:v2
#dict:
  $0=active
  $1~=https://api.example.com/v2/users/
#end
```

`=` declares an exact match; `~=` declares a URL prefix. In the body, `$0` replaces
exact values and `{$1}42` expands to the prefix + suffix. Keys are never replaced,
only values.

The dictionary is omitted when the total token savings from all entries do not exceed
the header/footer overhead (12 tokens) plus a per-entry minimum of 2 tokens net. This
prevents regressions on documents with few repeated strings.

### Grammar sketch

```
document  ::= ("#vdf:v2" CRLF "#dict:" CRLF entry* "#end" CRLF)? body
entry     ::= "  " "$" digits ("=" | "~=") repr CRLF
body      ::= value
value     ::= primitive | object | array
object    ::= (key "=" prim_val CRLF)*
            | (key ":" CRLF TAB+ value)*
array     ::= "[0]"
            | "[" N "] " comma_vals
            | "#" N " " (key " ")* CRLF (TAB row CRLF)*
            | "[" N "]" CRLF (TAB "- " item CRLF)*
prim_val  ::= bare_string | quoted_string | "null" | "true" | "false" | number
            | "$" digits | "{$" digits "}" suffix
```

### Worked example — nested uniform array

Input (`nested_uniform.json`, abbreviated):
```json
[{"id":1,"address":{"city":"Moscow","zip":"101000"},"score":98.5,"active":true}, ...]
```

JTF v2 output:
```
#20 id name address.city address.zip score active
    1    Alice Johnson    Moscow    101000    98.5    true
    2    Bob Smith    Saint Petersburg    190000    87.2    true
    ...
```

Dotted-path header `address.city` and `address.zip` eliminate the nested key names
entirely. The decoder splits on `.` to reconstruct `{"address": {"city": ..., "zip": ...}}`.

### Worked example — value dictionary

Input (`repeated_values.json`, 15 rows with repeated `status`, `plan`, `region`,
`created`, and URL fields):

```
#vdf:v2
#dict:
  $0=eu-west-1
  $1=active
  $2=premium
  $3=gold
  $4="2024-11-18T00:00:00Z"
  $5~=https://api.example.com/v2/users/
  $6=basic
  $7=silver
  $8=pending
  $9=us-east-1
  $10=bronze
  $11=suspended
  $12=ap-south-1
  $13=enterprise
  $14=platinum
#end
#15 id status plan region url created tier
    1    $1    $2    $0    {$5}1    $4    $3
    2    $8    $6    $0    {$5}2    $4    $7
    ...
```

## Benchmark (tiktoken cl100k_base, measured)

| file | JSON compact | JTF v1 | JTF v2 | v2 vs JSON | v2 vs v1 |
|------|-------------|--------|--------|-----------|---------|
| config.json | 133 | 154 | 133 | +0.0% | +13.6% |
| users_table.json | 294 | 215 | 214 | +27.2% | +0.5% |
| api_response.json | 289 | 303 | 299 | -3.5% | +1.3% |
| logs.json | 556 | 448 | 399 | +28.2% | +10.9% |
| repeated_values.json | 794 | 623 | 364 | +54.2% | +41.6% |
| nested_uniform.json | 759 | 990 | 473 | +37.7% | +52.2% |
| **TOTAL** | **2825** | **2733** | **1882** | **+33.4%** | **+31.1%** |

Positive % = tokens saved vs that baseline. Negative % = regression.

`api_response.json` is the one case where JTF v2 is worse than compact JSON (-3.5%).
It is a small document with unique keys and URLs, deep nesting, and no repeated
string values large enough to warrant a dictionary. The format cannot compress it
below the information content that compact JSON already achieves. JTF v2 is better
than JTF v1 on this file (+1.3%).

`config.json` breaks even at exactly 0% vs compact JSON, improved from -15.8% in
JTF v1 thanks to the `key=value` separator change.

`nested_uniform.json` shows the largest gain for a single file: 37.7% vs compact
JSON, with nested tabular encoding reducing 759 tokens to 473.

## Installation

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```sh
python3 jtf.py encode file.json [-o output.jtf]
python3 jtf.py decode file.jtf  [-o output.json]
python3 jtf.py bench  file.json
```

`encode` reads a JSON file and writes JTF to stdout (or `-o file`).

`decode` reads a JTF file and writes pretty-printed JSON to stdout (or `-o file`).

`bench` prints a table comparing token counts for compact JSON, JTF v1, and JTF v2
for a single file.

## Running tests

```sh
venv/bin/python tests/test_roundtrip.py   # 76 tests, all pass
venv/bin/python tests/benchmark.py        # prints the table above
```

Round-trip guarantee: `decode(encode(x)) == x` for any JSON-serializable Python
object. 76 test cases cover: all sample files, null/true/false/empty values,
numeric-looking strings, strings with every special character, deeply nested
structures (5 levels), arrays of arrays, mixed arrays, nested-uniform arrays,
and dictionary boundary cases.

## API

```python
from jtf import encode, decode

text = encode(data)   # Python object -> JTF string
obj  = decode(text)   # JTF string -> Python object
```

## When JTF v2 helps and when it does not

It helps most when:
- Data is an array of objects with the same schema (logs, DB results, API lists).
- String values repeat across rows (statuses, regions, categories, dates).
- Arrays of nested-uniform objects (user records with addresses, etc.).

It does not help (may regress) when:
- The document is small with mostly unique values and deep nesting (`api_response.json`).
- All values are large unique strings (there is nothing to factor out).
- Every key appears exactly once (the key=value separator saves 1 token, but this
  is offset by the absence of JSON's structural compactness).

Compact JSON is still a strong baseline. JTF v2 is a prompt-injection tool, not a
compression format — it is most useful when the LLM needs to read and understand the
data, not just store bytes.
