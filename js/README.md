# @jtf/jtf (JavaScript)

JavaScript (ESM) implementation of **JTF — JSON Token Format**, a lossless,
token-efficient encoding of the JSON data model for LLM prompts. Implements
both `encode` and `decode`.

See the [project SPEC.md](../SPEC.md) for the authoritative format
specification and the [top-level README](../README.md) for an overview.

## Use

```js
import { encode, decode } from "@jtf/jtf";

const text = encode({ id: 1, name: "Ada" }); // object -> JTF
const obj = decode(text); // JTF -> object
// obj deep-equals { id: 1, name: "Ada" } — round-trip is lossless
```

The encoder's value dictionary uses a deterministic heuristic cost function by
default (the conformance contract). Pass your own tokenizer-backed cost
function to optimize for a real model:

```js
import { encode } from "@jtf/jtf";
const text = encode(data, (s) => myTokenizer.encode(s).length);
```

## CLI

```sh
node bin/jtf.js encode input.json -o output.jtf
node bin/jtf.js decode output.jtf  -o roundtrip.json
```

## Tests

```sh
node --test
```

> Note: the package name `@jtf/jtf` is illustrative; the module is consumed
> directly from source in this monorepo and is not published to npm.
