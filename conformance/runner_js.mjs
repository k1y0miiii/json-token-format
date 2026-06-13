// JS side of the conformance runner.
//
// For every case in manifest.json, verifies BOTH directions against the
// golden vectors:
//   encode(input)   === expected   (.jtf is the contract)
//   decode(expected) deep-equals input
//
// Emits one JSON object per line: {id, name, encode_ok, decode_ok, ...}.
// Exit code is non-zero if any case fails.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import assert from "node:assert";
import { encode, decode } from "../js/src/jtf.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CASES = path.join(HERE, "cases");
const manifest = JSON.parse(
  readFileSync(path.join(HERE, "manifest.json"), "utf8")
);

// A golden .jtf file always ends with a trailing newline we add on write;
// encode() returns no trailing newline, so compare against the trimmed form.
function readExpected(file) {
  const raw = readFileSync(path.join(CASES, file), "utf8");
  return raw.endsWith("\n") ? raw.slice(0, -1) : raw;
}

let pass = 0;
let fail = 0;
const results = [];

for (const c of manifest.cases) {
  const input = JSON.parse(readFileSync(path.join(CASES, c.input), "utf8"));
  const expected = readExpected(c.expected);
  const r = { id: c.id, name: c.name, encode_ok: false, decode_ok: false };
  try {
    const got = encode(input);
    r.encode_ok = got === expected;
    if (!r.encode_ok) {
      r.encode_diff = { got, expected };
    }
    const decoded = decode(expected);
    try {
      assert.deepStrictEqual(decoded, input);
      r.decode_ok = true;
    } catch {
      r.decode_ok = false;
      r.decode_got = decoded;
    }
  } catch (e) {
    r.error = String(e && e.stack ? e.stack : e);
  }
  if (r.encode_ok && r.decode_ok) pass++;
  else fail++;
  results.push(r);
}

for (const r of results) console.log(JSON.stringify(r));
console.error(`JS: ${pass}/${manifest.cases.length} passed, ${fail} failed`);
process.exit(fail === 0 ? 0 : 1);
