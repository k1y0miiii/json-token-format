// Round-trip and behavioural tests for the JS JTF module (node --test).

import { test } from "node:test";
import assert from "node:assert/strict";
import { encode, decode } from "../src/jtf.js";

function chr(i) {
  return String.fromCharCode(i);
}

const ROUNDTRIP_CASES = [
  // primitives
  ["null", null],
  ["true", true],
  ["false", false],
  ["int zero", 0],
  ["negative int", -42],
  ["float", 3.14],
  ["negative float", -0.5],
  ["exp float", 1.5e10],
  // numeric-looking strings (stay strings)
  ["num-str 007", "007"],
  ["num-str 3.14", "3.14"],
  ["num-str 1e5", "1e5"],
  ["num-str true", { flag: "true" }],
  ["num-str false", { flag: "false" }],
  ["num-str null", { x: "null" }],
  ["num-str -1", { "-1": "value", k: "-1" }],
  ["num-str 0.0", "0.0"],
  ["num-str 1E10", "1E10"],
  // empty values
  ["empty string", ""],
  ["empty object", {}],
  ["empty array", []],
  ["empty obj val", { a: {} }],
  ["empty arr val", { a: [] }],
  ["empty nested", { a: { b: {} } }],
  ["empty string value", { k: "" }],
  // special-character strings
  ["string with =", "key=value"],
  ["string with :", "key: value"],
  ["string with tab", "a\tb"],
  ["string with newline", "line1\nline2"],
  ["string with comma", "one, two, three"],
  ["string with quote", 'say "hello"'],
  ["string with backslash", "C:\\Users\\test"],
  ["string starting with #", "#hashtag"],
  ["string starting with $", "$money"],
  ["string starting with @", "@handle"],
  ["string with equals at end", { k: "val=" }],
  ["string with pipe", "a|b|c"],
  ["string with brackets", "arr[0]"],
  ["string with braces", "{obj}"],
  ["cyrillic", "Привет мир"],
  ["emoji", "test☃"],
  ["mixed cyrillic emoji", { msg: "Привет 😀" }],
  ["leading space string", { k: " leading" }],
  ["trailing space string", { k: "trailing " }],
  // objects
  ["flat object", { name: "Иван", age: 30, active: true, score: 9.5, notes: null }],
  ["nested object", { app: "test", db: { host: "localhost", port: 5432, pool: { min: 2, max: 10 } } }],
  ["deeply nested 4", { l1: { l2: { l3: { l4: { value: "deep", n: 42 } } } } }],
  ["deeply nested 5", { a: { b: { c: { d: { e: "bottom" } } } } }],
  ["reserved-word keys", { null: 1, true: 2, false: 3 }],
  ["= in key", { "ke=y": "val" }],
  [": in key", { "ke:y": "val" }],
  ["tab in key", { "ke\ty": "val" }],
  ["cyrillic key", { имя: "Иван" }],
  // arrays
  ["primitive int array", [1, 2, 3, 4, 5]],
  ["primitive string array", ["alpha", "beta", "gamma"]],
  ["mixed primitive array", [1, "two", null, true, 3.14]],
  ["array of arrays", [[1, 2], [3, 4], [5, 6]]],
  ["array of arrays nested", [[1, [2, 3]], [4, [5, 6]]]],
  ["mixed/heterogeneous array", [{ a: 1 }, { a: 1, b: 2 }, "plain string", 42, null]],
  ["uniform flat array", [
    { id: 1, name: "Alice", active: true },
    { id: 2, name: "Bob", active: false },
    { id: 3, name: "Клод", active: true },
  ]],
  ["uniform array with nulls", [
    { x: 1, y: null, z: "a" },
    { x: 2, y: 99, z: "b" },
  ]],
  ["uniform array cyrillic keys", [
    { id: 1, имя: "Алексей", роль: "admin" },
    { id: 2, имя: "Мария", роль: "editor" },
  ]],
  ["nested uniform array", [
    { id: 1, addr: { city: "Moscow", zip: "101000" } },
    { id: 2, addr: { city: "Kazan", zip: "420000" } },
    { id: 3, addr: { city: "Samara", zip: "443000" } },
  ]],
  ["array of empty objects", [{}, {}, {}]],
  ["single-element array", [42]],
  ["single-element obj array", [{ x: 1 }]],
  ["key with space tabular", [
    { "first name": "Ann", age: 1 },
    { "first name": "Bob", age: 2 },
  ]],
  // nested containers in objects
  ["nested array in object", { users: [{ id: 1, name: "Alice" }, { id: 2, name: "Bob" }], count: 2 }],
  ["array of arrays in object", { matrix: [[1, 0], [0, 1]] }],
  ["url in value", { url: "https://example.com/path?q=1&r=2", title: "Example" }],
  ["timestamps", { created: "2024-11-18T14:32:07Z", updated: "2024-11-19T00:00:00Z" }],
  // dictionary boundary cases
  ["dict value with = sign", [
    { status: "key=value", other: "key=value" },
    { status: "key=value", other: "different" },
  ]],
  ["dict value that looks numeric", [
    { v: "3.14", other: "3.14" },
    { v: "3.14", other: "something" },
  ]],
  ["no dict when not profitable", { x: "ab", y: "ab" }],
  // top-level primitives
  ["top-level string", "hello world"],
  ["top-level int", 42],
  ["top-level null", null],
  ["top-level bool", true],
  ["top-level float", 3.14],
];

// build the $1-vs-$10 case programmatically
{
  const o = {};
  for (let i = 0; i < 11; i++) {
    const c = chr(97 + i);
    const v = `${c}${c}${c}${c}_${c}${c}${c}${c}_${c}${c}${c}${c}_${c}${c}${c}${c}`;
    o[`v${i}`] = v;
    o[`r${i}`] = v;
  }
  ROUNDTRIP_CASES.push(["dict boundary $1 vs $10", o]);
}

for (const [name, value] of ROUNDTRIP_CASES) {
  test(`roundtrip: ${name}`, () => {
    const encoded = encode(value);
    assert.equal(typeof encoded, "string");
    const decoded = decode(encoded);
    assert.deepEqual(decoded, value);
  });
}

// ---- output-shape assertions ----------------------------------------------

test("kv separator is equals", () => {
  assert.equal(encode({ a: 1 }), "a=1");
});

test("nested block uses colon and tab", () => {
  assert.equal(encode({ a: { b: 1 } }), "a:\n\tb=1");
});

test("primitive array inline", () => {
  assert.equal(encode([1, 2, 3]), "[3] 1,2,3");
});

test("empty containers", () => {
  assert.equal(encode({}), "{}");
  assert.equal(encode([]), "[0]");
  assert.equal(encode({ a: {} }), "a={}");
  assert.equal(encode({ a: [] }), "a=[0]");
});

test("tabular header flat", () => {
  const out = encode([{ id: 1, name: "A" }, { id: 2, name: "B" }]);
  const lines = out.split("\n");
  assert.equal(lines[0], "#2 id name");
  assert.equal(lines[1], "\t1\tA");
  assert.equal(lines[2], "\t2\tB");
});

test("tabular header nested dotted", () => {
  const out = encode([
    { id: 1, addr: { city: "Moscow" } },
    { id: 2, addr: { city: "Kazan" } },
  ]);
  assert.equal(out.split("\n")[0], "#2 id addr.city");
});

test("numeric string quoted", () => {
  assert.equal(encode("007"), '"007"');
  assert.equal(encode("3.14"), '"3.14"');
});

test("reserved word string quoted", () => {
  assert.equal(encode("true"), '"true"');
  assert.equal(encode("null"), '"null"');
});

test("dictionary emitted for repeated values", () => {
  const data = [];
  for (let i = 0; i < 10; i++) data.push({ s: "a_repeated_long_value_here" });
  const out = encode(data);
  assert.ok(out.startsWith("#vdf:v2\n#dict:\n"));
  assert.ok(out.includes("#end"));
  assert.deepEqual(decode(out), data);
});

test("no dictionary when unprofitable", () => {
  const out = encode({ x: "ab", y: "ab" });
  assert.ok(!out.includes("#vdf"));
});
