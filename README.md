<div align="center">

# JTF — JSON Token Format

**English** · [Русский](README.ru.md)

A lossless, round-trippable encoding of the JSON data model that cuts the number
of tokens an LLM spends on structured data — **−33% versus compact JSON**
(measured with `tiktoken` `cl100k_base`).

</div>

---

JTF is not a compression format. Its output is human-readable UTF-8 text that a
language model reads and reasons about directly. It wins by dropping JSON's
redundant punctuation, turning arrays of uniform records into a compact table,
and factoring out repeated string values — while guaranteeing
`decode(encode(x)) == x` for every JSON value.

This repository matures JTF into a **standard**:

- **[SPEC.md](SPEC.md)** — the authoritative, implementation-independent
  specification. You can write a conformant encoder/decoder from it alone.
- **[conformance/](conformance/)** — language-agnostic golden vectors that
  *define* compliance. Both libraries pass all of them.
- **[python/](python/)** — reference Python package (`jtf`): encode, decode, CLI.
- **[js/](js/)** — JavaScript (ESM) module: encode and decode.

## What it looks like

JSON:

```json
[
  {"id": 1, "name": "Alice", "active": true},
  {"id": 2, "name": "Bob",   "active": false}
]
```

JTF:

```
#2 id name active
	1	Alice	true
	2	Bob	false
```

The header names the columns once; each row is just tab-separated cells. Nested
records collapse to dotted paths (`addr.city`), and repeated string values are
hoisted into a small `$N` dictionary when that saves tokens. See
[SPEC.md](SPEC.md) for every rule.

## Install & use

### Python

```sh
pip install -e python/            # or: pip install -e "python/[tiktoken]"
```

```python
from jtf import encode, decode

text = encode({"id": 1, "name": "Ada"})
obj  = decode(text)               # == {"id": 1, "name": "Ada"}
```

CLI:

```sh
jtf encode data.json -o data.jtf
jtf decode data.jtf  -o roundtrip.json
jtf encode data.json --tiktoken   # tune the dictionary to a real tokenizer
```

### JavaScript (Node ≥ 18, ESM)

```js
import { encode, decode } from "./js/src/jtf.js";

const text = encode({ id: 1, name: "Ada" });
const obj  = decode(text);        // deep-equals { id: 1, name: "Ada" }
```

CLI:

```sh
node js/bin/jtf.js encode data.json -o data.jtf
node js/bin/jtf.js decode data.jtf  -o roundtrip.json
```

## The conformance idea

A format is only a *standard* if independent implementations agree byte-for-byte.
`conformance/` holds `NNN-name.json` inputs paired with their expected
`NNN-name.jtf` outputs (indexed by `manifest.json`). For every vector, a
conformant library must (1) re-encode the input to exactly the expected `.jtf`,
and (2) decode the `.jtf` back to exactly the input. Python is the source of
truth; JavaScript is aligned to it; the vectors are the contract.

```sh
./run-conformance.sh
```

```
JTF conformance suite — 45 vectors
================================================
Python : 45/45 passed
JS     : 45/45 passed
================================================
RESULT : PASS — both languages pass all 45 vectors.
```

## Token result (measured)

`tiktoken` `cl100k_base`, JTF (production mode) vs compact JSON, on the six
reference documents:

| file | JSON compact | JTF | saved |
|------|-------------:|----:|------:|
| config.json | 133 | 133 | 0.0% |
| users_table.json | 294 | 214 | 27.2% |
| api_response.json | 289 | 299 | −3.5% |
| logs.json | 556 | 399 | 28.2% |
| repeated_values.json | 794 | 364 | 54.2% |
| nested_uniform.json | 759 | 473 | 37.7% |
| **TOTAL** | **2825** | **1882** | **33.4%** |

JTF helps most for arrays of uniform records, repeated string values, and nested
records; it can regress slightly on small all-unique documents. The number is
reproduced from the origin project, not re-invented.

## Tests

```sh
cd python && pytest          # Python unit tests
cd js && node --test         # JS unit tests
./run-conformance.sh         # shared conformance suite (both languages)
```

## Roadmap

- **Rust port** (`rust/`) — planned, conformance-driven against the same vectors.
- **Go port** (`go/`) — planned, same contract.
- Streaming encode/decode for very large arrays.

Ports are not built yet; the vectors in `conformance/` will be their acceptance
test, exactly as for Python and JS today.

## Related projects

- _JTF originally shipped as **json-token-format** — now consolidated into this repo as its canonical home (old links redirect here)._
- **token-diet** — interactive playground visualizing JTF savings against live
  GPT tokenizers.
- **llmcost** — token/cost tooling for LLM prompts.
- **llm-gateway** — token-frugal LLM proxy that uses JTF to compress payloads.

## License

[MIT](LICENSE) © 2026 Maxim Chumakov
