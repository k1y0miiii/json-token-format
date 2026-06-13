# JTF — JSON Token Format · Specification v1.0

> Status: stable. This document is the authoritative, implementation-independent
> specification of JTF. A conformant encoder and decoder can be written from this
> document alone. The reference implementation lives in [`python/`](python/); the
> machine-checkable contract lives in [`conformance/`](conformance/).

## 1. Purpose and scope

JTF is a **lossless, round-trippable text encoding of the JSON data model**,
designed to reduce the number of tokens a large-language-model tokenizer spends
on structured data placed in a prompt. It is not a binary or compression format:
the output is human-readable UTF-8 text that an LLM can read and reason about
directly.

The design target is **`decode(encode(x)) == x`** for every value `x` in the JSON
data model, while emitting fewer tokens than compact JSON for the common shapes
of LLM-facing data (arrays of uniform records, repeated string values, nested
records).

Measured result: on the project's six reference documents, JTF produces
**−33.4% tokens versus compact JSON** under OpenAI's `tiktoken` `cl100k_base`
tokenizer. See [§11](#11-token-result-measured) for the per-file table; the
number is reproduced, not re-invented, from the origin project
[`json-token-format`](#12-related-projects).

## 2. Data model

JTF encodes exactly the **JSON data model**:

| JSON type | Notes |
|-----------|-------|
| object    | An ordered collection of string-keyed members. Key order is preserved. Keys are assumed unique. |
| array     | An ordered sequence of values. |
| string    | A sequence of Unicode characters. |
| number    | An integer or IEEE-754 double, in canonical JSON form. |
| `true` / `false` | Boolean. |
| `null`    | The null value. |

A conformant implementation MUST preserve object **key order** and array
**element order**. It MUST preserve the distinction between a numeric value and a
string that happens to look numeric (e.g. the number `7` versus the string
`"007"`).

### 2.1 Number representation note (cross-language)

JTF writes numbers in canonical JSON form (the shortest decimal string that
round-trips the IEEE-754 double, the same form `JSON.stringify` / Python
`json.dumps` produce). One environment-level caveat applies to **integral
floats** such as `45.0`: a host language with a single numeric type (e.g.
JavaScript) collapses `45.0` to the integer `45` at JSON-parse time, before JTF
ever sees the value, so the trailing `.0` cannot be reproduced. This is a
property of the host's JSON parser, not of JTF. Implementations on such hosts are
conformant; the shared conformance vectors deliberately avoid integral-float
inputs so the contract is satisfiable in every language.

## 3. Document structure

```
document ::= [ dictionary-block ] body
```

A JTF document is UTF-8 text. Lines are separated by `\n` (LF). A conformant
decoder MUST also accept `\r\n`. The body is the top-level JSON value written at
indentation level 0 with **no enclosing braces or brackets** for the outermost
object/array.

Indentation is significant and uses **one TAB (`\t`) per nesting level**. Spaces
are never used for structural indentation. (The two-space dictionary entry prefix
in [§9](#9-value-dictionary) is the sole exception and is not structural.)

## 4. Primitive values

A primitive is `null`, a boolean, a number, or a string.

| Value | Encoding |
|-------|----------|
| `null` | `null` |
| `true` | `true` |
| `false` | `false` |
| number | canonical JSON form (e.g. `42`, `-7`, `3.14`, `0.1`, `-0.5`) |
| string | bare when *safe* (§5), otherwise a JSON double-quoted string |

### 4.1 Strings — safety rule

A string MAY be written **bare** (without surrounding quotes) if and only if all
of the following hold:

1. It is non-empty.
2. It is not one of the reserved words `null`, `true`, `false`.
3. It equals its own whitespace-stripped form (no leading or trailing
   whitespace).
4. Its first character is none of: `"` `[` `{` `-` `#` `=` `$` `@`.
5. It does not contain the substring `"= "` and does not end with `=`.
6. It contains **none** of these characters:
   `\n` `\r` `,` `=` `:` `|` `"` `\` `#` `[` `]` `{` `}` `\t` `$` `@`
7. Its bare form would **not** be decoded back as a number, boolean, or null
   (the *numeric-string* rule, §4.2).

A string that fails any condition MUST be written as a **JSON double-quoted
string** using standard JSON escape sequences (`\"`, `\\`, `\n`, `\t`, `\r`,
`\uXXXX`, …). Non-ASCII characters (Cyrillic, emoji, CJK) are emitted literally
as UTF-8, never `\u`-escaped (`ensure_ascii=False`).

### 4.2 Numeric-string rule

If a string's bare form would be parsed as a number by the decoder (§7) — i.e.
it matches an integer or float literal such as `007`, `3.14`, `1e5`, `-1` — it
MUST be double-quoted even if it otherwise passes §4.1. This prevents the lossy
round-trip `"007"` → `007` → `7`. The reserved words `null`/`true`/`false` as
string values are quoted for the same reason (condition 2).

## 5. Objects

An object is encoded as a sequence of member lines, each indented to the object's
nesting depth. There are two member forms:

```
key=value          ← primitive (or single-line array) value
key:               ← nested block follows on the next, more-indented lines
	child=...
```

- **`key=value`** is used when the value is a primitive, an empty container, or
  an array that renders to a single line ([§6.1](#61-primitive-arrays-inline),
  [§6.4](#64-empty-and-single-line-cases)). The `=` separator is chosen because
  it tokenizes to one token with no surrounding space, unlike JSON's `": "`.
- **`key:`** introduces a nested object or a multi-line array. The child content
  appears on the following lines, indented one level deeper.

The key is encoded by the string rules of §4.1/§4.2 (bare when safe, quoted
otherwise). An **empty object** value is written inline as `key={}`.

The **empty object** as a standalone value is `{}`.

### 5.1 Separator disambiguation

A decoder distinguishes the two forms by the first **unquoted** `=` or `:` in the
line. If an unquoted `=` occurs and precedes any unquoted `:` (or no `:`
exists), the line is `key=value`. Otherwise the first unquoted `:` separates a
`key:` nested block. Quoted keys/values shield their internal `=`/`:` from this
scan.

## 6. Arrays

The array form is selected by inspecting the elements, in this priority order:

1. **Empty** → §6.4
2. **All primitives** → inline, §6.1
3. **Uniform objects** (all elements are objects exposing the identical set of
   dotted leaf-paths, every leaf primitive) → tabular, §6.2
4. **Otherwise** → dash list, §6.3

### 6.1 Primitive arrays (inline)

When every element is a primitive, the array is one line:

```
[N] v1,v2,v3
```

`N` is the element count. Values are comma-separated and encoded per §4. Note the
single space after `]` and the absence of spaces around commas.

### 6.2 Tabular arrays (uniform objects)

An array qualifies as **tabular** when it is non-empty, every element is a
non-empty object, and all elements expose **exactly the same set of dotted
leaf-paths** with **all-primitive leaves**. A *dotted leaf-path* is formed by
walking nested objects and joining keys with `.` (e.g. `addr.city`). A leaf that
is an array disqualifies tabular encoding.

Encoding:

```
#N path1 path2 path3
	v1	v2	v3
	v4	v5	v6
```

- The header line is `#`, the count `N`, a single space, then the leaf-paths in
  first-element order, **space-separated**.
- Each subsequent line is one record, indented one level deeper than the header,
  with cells **TAB-separated** and encoded per §4.
- Paths are taken from the **first** element's key order; every element MUST
  supply the same path set.

The flat case (no nested objects) is just the dotted case with no `.` in any
path, e.g. `#3 id name active`.

**Unsafe or spaced path names.** If any path is not §4.1-safe or contains a space
(which would break the space-separated header), the header switches to CSV form:

```
#N csv:"path one","path two"
	v1	v2
```

After `csv:`, paths are comma-separated and individually bare-or-quoted per §4.1;
a decoder splits this list respecting double-quoted fields. Rows remain
TAB-separated regardless.

On decode, each cell is assigned to its path; dotted paths are split on `.` to
rebuild the nested object structure.

### 6.3 Mixed / heterogeneous arrays (dash list)

Any array that is neither all-primitive nor uniform-tabular uses a dash list:

```
[N]
	- item0
	- item1
```

- The header is `[N]` alone on its line.
- Each item begins with `\t- ` at one level deeper than the header.
- A primitive item is written inline after the dash.
- A container item is written with its **first** line after the `- ` and its
  remaining lines indented **two** levels deeper than the array header (so the
  decoder reads them as the item's children). An empty item value is `\t-` with
  nothing after the dash, decoding to `null` only when the source element was
  `null`; an empty object/array element is encoded by its own rule (`{}` /
  `[0]`).

### 6.4 Empty and single-line cases

| Value | Encoding |
|-------|----------|
| `[]` (empty array) | `[0]` |
| `{}` (empty object) | `{}` |
| empty array as object value | `key=[0]` |
| empty object as object value | `key={}` |

## 7. Decoding primitives

A bare token decodes as follows, in order:

1. `null` → null, `true` → true, `false` → false.
2. `{}` → empty object, `[]` → empty array.
3. A token beginning with `"` → JSON-parse it as a string.
4. Else attempt a number: if the token contains `.` or `e`/`E`, parse as float;
   otherwise parse as integer. (Conformant parsers accept the canonical JSON
   numeric grammar.)
5. Otherwise the token is a bare string, taken verbatim.

Dictionary tokens (`$N`, `{$N}suffix`, §9) are resolved **before** steps 1–5.

## 8. Indentation and whitespace rules (normative summary)

- Structural indentation is TAB only, one per level.
- A line's nesting depth is its count of leading TAB characters.
- Blank lines are insignificant and skipped by the decoder.
- The encoder emits no trailing whitespace on any line and no trailing newline on
  the document as a whole (a single trailing newline, as produced by file
  writers/CLIs, is accepted by the decoder).

## 9. Value dictionary

When the same string **values** repeat across a document and substituting short
tokens for them saves tokens net of the dictionary's own cost, JTF prepends a
**value-dictionary block**:

```
#vdf:v2
#dict:
  $0=active
  $1~=https://api.example.com/v2/users/
#end
```

- The block is delimited by the literal lines `#vdf:v2`, `#dict:`, and `#end`.
- Each entry is `  $N` (two leading spaces), then a separator, then the value
  representation (bare-or-quoted per §4.1):
  - `=` declares an **exact** value alias. In the body, `$N` stands for the whole
    value.
  - `~=` declares a **prefix** alias. In the body, `{$N}suffix` expands to
    `value + suffix`.
- Tokens are numbered `$0`, `$1`, … in admission order. `$10` is distinct from
  `$1`; decoders MUST match the full `$\d+` token, and `{$N}` is delimited by the
  closing `}` so suffixes are unambiguous.
- **Only string values are ever substituted. Keys are never substituted.**

### 9.1 Break-even logic (informative but precise)

The dictionary is built so it never increases token count. Implementations
estimate token cost with a **cost function** `cost(s) → int`:

- **Canonical / conformance cost function:** the deterministic, dependency-free
  estimate `cost(s) = max(1, floor(len(s) / 4))`. This is what `encode` uses by
  default and what the conformance vectors are generated with, so output is
  byte-identical across languages and machines.
- **Production cost function:** a real tokenizer (e.g. `tiktoken` `cl100k_base`).
  This yields the best real-world savings but may make different borderline
  dictionary decisions, so its output is not part of the cross-language contract.

The build proceeds in two phases:

1. **Exact entries.** Candidate values are those occurring ≥ 2 times, processed
   most-valuable-first (by `count × cost(repr)`). A candidate is admitted only if
   `cost(repr) > cost("$N")` and its gross savings
   `(cost(repr) − cost("$N")) × count` exceed `cost(dict_line) + 2`
   (a per-entry net-savings guard of 2).
2. **Prefix entries.** URL prefixes (`http(s)://…/`, length ≥ 10) and ISO-8601
   date prefixes (`YYYY-MM-DDT`) occurring ≥ 2 times are considered. A prefix is
   admitted if it covers ≥ 2 still-unaliased values and its gross savings,
   computed against the `{$N}suffix` body form, clear the same per-entry guard.

After both phases, a **global check** discards the entire dictionary if the
combined net savings do not exceed the fixed header/footer overhead
(`cost("#vdf:v2\n#dict:\n") + cost("#end\n")`). This prevents regressions on
documents with only a few cheap substitutions.

When the dictionary is empty, the document is just the body — no `#vdf:v2` line.

## 10. Grammar (sketch)

```
document   ::= dictionary? body
dictionary ::= "#vdf:v2" LF "#dict:" LF entry* "#end" LF
entry      ::= "  " "$" DIGITS ("=" | "~=") value-repr LF
body       ::= value
value      ::= primitive | object | array
object     ::= ( key "=" prim-or-inline LF
               | key ":" LF INDENT value DEDENT
               | key "={}" LF )*
array      ::= "[0]"
             | "[" N "] " comma-values
             | "#" N " " header LF ( TAB row LF )*
             | "[" N "]" LF ( TAB "- " item )*
header     ::= path ( " " path )* | "csv:" csv-paths
row        ::= cell ( TAB cell )*
primitive  ::= "null" | "true" | "false" | number
             | bare-string | quoted-string
             | "$" DIGITS | "{$" DIGITS "}" suffix
```

Indentation (`INDENT`/`DEDENT`) is carried by leading TABs, one per level.

## 11. Token result (measured)

`tiktoken` `cl100k_base`, JTF in production mode (tiktoken-tuned dictionary),
versus compact JSON (`separators=(',',':')`), on the six reference documents:

| file | JSON compact | JTF | saved |
|------|-------------:|----:|------:|
| config.json | 133 | 133 | 0.0% |
| users_table.json | 294 | 214 | 27.2% |
| api_response.json | 289 | 299 | −3.5% |
| logs.json | 556 | 399 | 28.2% |
| repeated_values.json | 794 | 364 | 54.2% |
| nested_uniform.json | 759 | 473 | 37.7% |
| **TOTAL** | **2825** | **1882** | **33.4%** |

Positive = tokens saved. JTF helps most for arrays of uniform records, repeated
string values, and nested-uniform records. It can regress slightly on small
documents with all-unique values and deep nesting (`api_response.json`), where
compact JSON is already near the information floor. Compact JSON remains a strong
baseline; JTF is a prompt-shaping tool, not a byte-compression format.

This table is reproduced from the origin project, not re-derived — the refactored
reference implementation in [`python/`](python/) regenerates it exactly.

## 12. Related projects

- _JTF originally shipped as **json-token-format** — now consolidated into this repo._
- **token-diet** — interactive playground that visualizes JTF's token savings
  against live GPT tokenizers; source of the faithful JS encode port.
- **llmcost** — token/cost tooling for LLM prompts.

## 13. Conformance

An implementation is **JTF-conformant** if, for every vector in
[`conformance/`](conformance/) (each a `NNN-name.json` input with its
`NNN-name.jtf` expected output, indexed by `conformance/manifest.json`):

1. `encode(input)` equals the expected `.jtf` byte-for-byte (using the canonical
   cost function of §9.1), and
2. `decode(expected)` equals `input`.

The reference Python and JS libraries both pass all vectors; run
`./run-conformance.sh` (or `python3 conformance/run.py`) to verify.
