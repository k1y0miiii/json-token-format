// ===========================================================================
// JTF — JSON Token Format (JavaScript, ESM)
//
// A lossless, round-trippable encoding of the JSON data model that reduces LLM
// tokenizer token count. This module implements BOTH encode and decode.
//
// The encoder is a faithful port of the reference Python implementation
// (python/src/jtf/core.py), which is the source of truth. The value-dictionary
// break-even analysis uses a pluggable, deterministic cost function so that
// JS and Python produce byte-identical output for the conformance suite.
//
// See SPEC.md for the authoritative format specification.
// ===========================================================================

import { heuristicCost } from "./cost.js";

// ---- string safety --------------------------------------------------------

const JTF_RESERVED = new Set(["null", "true", "false"]);
// Forbidden chars in a bare string: \n \r , = : | " \ # [ ] { } \t $ @
const JTF_UNSAFE_CHARS = /[\n\r,=:|"\\#\[\]{}\t$@]/;

function jtfSafe(s) {
  if (typeof s !== "string" || s.length === 0) return false;
  if (JTF_RESERVED.has(s)) return false;
  if (s !== s.trim()) return false;
  if ('"[{-#=$@'.includes(s[0])) return false;
  if (s.includes("= ") || s.endsWith("=")) return false;
  return !JTF_UNSAFE_CHARS.test(s);
}

// Would a bare string be mis-read as a number/bool on decode?
function jtfLooksNumeric(s) {
  if (typeof s !== "string" || s.trim() === "") return false;
  const t = s.trim();
  if (t !== s) return false;
  if (/[.eE]/.test(t)) return /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t);
  return /^[+-]?\d+$/.test(t);
}

// JSON-string form matching Python json.dumps(s, ensure_ascii=False).
// JS JSON.stringify produces compatible escapes for the relevant cases.
const jsonStr = (s) => JSON.stringify(s);

// Canonical number form. Python json.dumps and JS produce identical output for
// every value that survives a JSON.parse round-trip (JSON has no NaN/Inf, and
// integral floats like 5.0 collapse to 5 in both after parsing).
function jtfNumber(n) {
  return String(n);
}

// Encode a primitive value (mirrors core._ep).
function jtfEncPrim(v) {
  if (v === null) return "null";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return jtfNumber(v);
  if (jtfLooksNumeric(v)) return jsonStr(v);
  return jtfSafe(v) ? v : jsonStr(v);
}

const jtfEs = (s) => (jtfSafe(s) ? s : jsonStr(s));
const jtfValRepr = jtfEs;

const isPrim = (v) =>
  v === null ||
  typeof v === "boolean" ||
  typeof v === "number" ||
  typeof v === "string";
const isPlainObject = (v) =>
  v !== null && typeof v === "object" && !Array.isArray(v);

// ---- nested-uniform detection (core._is_nested_uniform) -------------------

function dottedPathsOrdered(obj, prefix = "") {
  const result = [];
  for (const k of Object.keys(obj)) {
    const v = obj[k];
    const path = prefix ? `${prefix}.${k}` : k;
    if (isPrim(v)) result.push(path);
    else if (isPlainObject(v)) {
      const child = dottedPathsOrdered(v, path);
      if (child === null) return null;
      result.push(...child);
    } else return null;
  }
  return result;
}

function isNestedUniform(arr) {
  if (!arr.length || !isPlainObject(arr[0]) || Object.keys(arr[0]).length === 0)
    return null;
  const paths0 = dottedPathsOrdered(arr[0]);
  if (paths0 === null || paths0.length === 0) return null;
  const set0 = new Set(paths0);
  for (let i = 1; i < arr.length; i++) {
    const item = arr[i];
    if (!isPlainObject(item)) return null;
    const ps = dottedPathsOrdered(item);
    if (ps === null) return null;
    const s = new Set(ps);
    if (s.size !== set0.size) return null;
    for (const p of s) if (!set0.has(p)) return null;
  }
  const hasDots = paths0.some((p) => p.includes("."));
  return { paths: paths0, hasDots };
}

function getPath(obj, path) {
  let cur = obj;
  for (const p of path.split(".")) cur = cur[p];
  return cur;
}

function setPath(obj, path, value) {
  const parts = path.split(".");
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (!(p in cur)) cur[p] = {};
    cur = cur[p];
  }
  cur[parts[parts.length - 1]] = value;
}

// ---- value dictionary (core._build_dictionary) ----------------------------

const PREFIX_MIN_LEN = 10;

function collectValues(data, vctr, pctr) {
  if (typeof data === "string") {
    vctr.set(data, (vctr.get(data) || 0) + 1);
    notePrefix(data, pctr);
  } else if (Array.isArray(data)) {
    for (const x of data) collectValues(x, vctr, pctr);
  } else if (isPlainObject(data)) {
    for (const v of Object.values(data)) collectValues(v, vctr, pctr);
  }
}

function notePrefix(s, pctr) {
  if (s.startsWith("http://") || s.startsWith("https://")) {
    const idx = s.lastIndexOf("/");
    if (idx > 8 && idx < s.length - 1) {
      const prefix = s.slice(0, idx + 1);
      if (prefix.length >= PREFIX_MIN_LEN)
        pctr.set(prefix, (pctr.get(prefix) || 0) + 1);
    }
  }
  if (s.length > 10 && s[10] === "T" && /^\d{4}-\d{2}-\d{2}T/.test(s)) {
    const p = s.slice(0, 11);
    pctr.set(p, (pctr.get(p) || 0) + 1);
  }
}

function dictHeaderOverhead(cost) {
  return cost("#vdf:v2\n#dict:\n") + cost("#end\n");
}

function buildDictionary(data, cost, minNetSavings = 2) {
  const vctr = new Map();
  const pctr = new Map();
  collectValues(data, vctr, pctr);

  const entries = [];
  const valueMap = new Map();
  const prefixCovered = new Set();
  let idx = 0;
  const entrySavings = [];
  const entryCosts = [];

  // Phase 1: exact values, most valuable first. Tie-break by Map insertion
  // order (matches Python's stable sort over Counter insertion order).
  const vItems = [...vctr.entries()].sort(
    (a, b) => b[1] * cost(jtfValRepr(b[0])) - a[1] * cost(jtfValRepr(a[0]))
  );
  for (const [val, count] of vItems) {
    if (count < 2) continue;
    const valRepr = jtfValRepr(val);
    const valCost = cost(valRepr);
    const tokName = `$${idx}`;
    const tokCost = cost(tokName);
    if (valCost <= tokCost) continue;
    const dictLine = `  ${tokName}=${valRepr}`;
    const entryCost = cost(dictLine);
    const gross = (valCost - tokCost) * count;
    if (gross > entryCost + minNetSavings) {
      entries.push({ token: tokName, value: val, isPrefix: false });
      valueMap.set(val, tokName);
      entrySavings.push(gross);
      entryCosts.push(entryCost);
      idx++;
    }
  }

  // Phase 2: URL / timestamp prefix entries (most common first).
  const pItems = [...pctr.entries()].sort((a, b) => b[1] - a[1]);
  for (const [prefix, pCount] of pItems) {
    if (pCount < 2) continue;
    const matching = [...vctr.keys()].filter(
      (v) => v.startsWith(prefix) && !valueMap.has(v)
    );
    if (matching.length < 2) continue;
    const prefixRepr = jtfValRepr(prefix);
    const tokName = `$${idx}`;
    const tokCost = cost(tokName);
    const dictLine = `  ${tokName}=${prefixRepr}`;
    const entryCost = cost(dictLine);
    let gross = 0;
    for (const v of matching) {
      const count = vctr.get(v);
      const fullCost = cost(jtfValRepr(v));
      const suffix = v.slice(prefix.length);
      const newCost = cost("{" + tokName + "}" + suffix);
      gross += (fullCost - newCost) * count;
    }
    if (gross > entryCost + minNetSavings) {
      entries.push({ token: tokName, value: prefix, isPrefix: true });
      for (const v of matching) prefixCovered.add(v);
      entrySavings.push(gross);
      entryCosts.push(entryCost);
      idx++;
    }
  }

  if (entries.length) {
    const fixed = dictHeaderOverhead(cost);
    const net =
      entrySavings.reduce((a, b) => a + b, 0) -
      entryCosts.reduce((a, b) => a + b, 0) -
      fixed;
    if (net <= 0)
      return { entries: [], valueMap: new Map(), prefixCovered: new Set() };
  }
  return { entries, valueMap, prefixCovered };
}

// ---- encoder core (core._Encoder) -----------------------------------------

function makeEncoder(valueMap, prefixCovered, prefixTokens) {
  const sortedPrefixes = [...prefixTokens].sort(
    (a, b) => b[0].length - a[0].length
  );
  function encStr(s) {
    if (valueMap.has(s)) return valueMap.get(s);
    if (prefixCovered.has(s)) {
      for (const [prefix, t] of sortedPrefixes) {
        if (s.startsWith(prefix)) return "{" + t + "}" + s.slice(prefix.length);
      }
    }
    return jtfEncPrim(s);
  }
  const encVal = (v) => (typeof v === "string" ? encStr(v) : jtfEncPrim(v));

  function encObj(obj, depth) {
    const keys = Object.keys(obj);
    if (keys.length === 0) return ["{}"];
    const pad = "\t".repeat(depth);
    const lines = [];
    for (const k of keys) {
      const v = obj[k];
      const ek = jtfEs(k);
      if (isPrim(v)) {
        lines.push(pad + `${ek}=${encVal(v)}`);
      } else if (Array.isArray(v)) {
        const arr = encArray(v, depth);
        if (arr.length === 1) lines.push(pad + `${ek}=${arr[0]}`);
        else {
          lines.push(pad + `${ek}:${arr[0]}`);
          for (let i = 1; i < arr.length; i++) lines.push(arr[i]);
        }
      } else if (isPlainObject(v)) {
        if (Object.keys(v).length === 0) lines.push(pad + ek + "={}");
        else {
          lines.push(pad + `${ek}:`);
          lines.push(...encObj(v, depth + 1));
        }
      } else {
        throw new TypeError(`Unsupported type for key ${k}`);
      }
    }
    return lines;
  }

  function encArray(arr, depth) {
    const pad = "\t".repeat(depth);
    const n = arr.length;
    if (n === 0) return ["[0]"];
    if (arr.every(isPrim)) {
      return [`[${n}] ` + arr.map(encVal).join(",")];
    }
    const uni = isNestedUniform(arr);
    if (uni !== null) {
      const { paths } = uni;
      const rowPad = pad + "\t";
      const safeKeys = paths.every((p) => jtfSafe(p) && !p.includes(" "));
      const hdr = safeKeys
        ? paths.join(" ")
        : "csv:" + paths.map(jtfEs).join(",");
      const lines = [`#${n} ${hdr}`];
      for (const item of arr) {
        lines.push(rowPad + paths.map((p) => encVal(getPath(item, p))).join("\t"));
      }
      return lines;
    }
    // mixed / dash list
    const contPad = pad + "\t\t";
    const lines = [`[${n}]`];
    for (const item of arr) {
      const il = encItem(item);
      if (il.length) {
        lines.push(pad + "\t- " + il[0]);
        for (let i = 1; i < il.length; i++) lines.push(contPad + il[i]);
      } else lines.push(pad + "\t-");
    }
    return lines;
  }

  function encItem(val) {
    if (isPrim(val)) return [encVal(val)];
    if (Array.isArray(val)) return encArray(val, 0);
    if (isPlainObject(val)) return encObj(val, 0);
    throw new TypeError("Unsupported item type");
  }

  return { encObj, encArray, encVal };
}

/**
 * Encode a JSON-compatible value to a JTF string.
 * @param {*} data Any JSON-serializable value.
 * @param {(s: string) => number} [cost] Token-cost estimator for the value
 *   dictionary. Defaults to the deterministic heuristic (the conformance
 *   contract). Pass a tokenizer-backed function to optimize for a real model.
 * @returns {string} JTF text (no trailing newline).
 */
export function encode(data, cost = heuristicCost) {
  const { entries, valueMap, prefixCovered } = buildDictionary(data, cost);
  const prefixTokens = entries
    .filter((e) => e.isPrefix)
    .map((e) => [e.value, e.token]);
  const enc = makeEncoder(valueMap, prefixCovered, prefixTokens);

  let bodyLines;
  if (isPlainObject(data)) bodyLines = enc.encObj(data, 0);
  else if (Array.isArray(data)) bodyLines = enc.encArray(data, 0);
  else bodyLines = [enc.encVal(data)];

  if (!entries.length) return bodyLines.join("\n");

  const header = ["#vdf:v2", "#dict:"];
  for (const e of entries) {
    const sep = e.isPrefix ? "~=" : "=";
    header.push(`  ${e.token}${sep}${jtfValRepr(e.value)}`);
  }
  header.push("#end");
  return header.join("\n") + "\n" + bodyLines.join("\n");
}

// ===========================================================================
// DECODER (inverse of encode; mirrors core._Decoder)
// ===========================================================================

const TAB_HDR = /^#(\d+) (.+)$/;
const ARR_HDR = /^\[(\d+)\](.*)$/;

function csvSplit(s) {
  const fields = [];
  let cur = "";
  let inQ = false;
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === '"' && !inQ) {
      inQ = true;
      cur += c;
    } else if (c === '"' && inQ) {
      if (i + 1 < s.length && s[i + 1] === '"') {
        cur += '"';
        i += 1;
      } else {
        inQ = false;
        cur += c;
      }
    } else if (c === "," && !inQ) {
      fields.push(cur);
      cur = "";
    } else {
      cur += c;
    }
    i += 1;
  }
  fields.push(cur);
  return fields;
}

function decStrTok(tok) {
  tok = tok.trim();
  if (tok.startsWith('"')) return JSON.parse(tok);
  return tok;
}

function decPrimBase(tok) {
  tok = tok.trim();
  if (tok === "null") return null;
  if (tok === "true") return true;
  if (tok === "false") return false;
  if (tok === "{}") return {};
  if (tok === "[]") return [];
  if (tok.startsWith('"')) return JSON.parse(tok);
  // number? Match Python int()/float() acceptance for canonical JSON forms.
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(tok)) {
    return Number(tok);
  }
  return tok;
}

function findEq(s) {
  let inQ = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '"' && (i === 0 || s[i - 1] !== "\\")) inQ = !inQ;
    if (!inQ && c === "=") return i;
  }
  return -1;
}

function findColon(s) {
  let inQ = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (c === '"' && (i === 0 || s[i - 1] !== "\\")) inQ = !inQ;
    if (!inQ && c === ":") return i;
  }
  return -1;
}

function looksLikeObjLine(content) {
  if (content.startsWith("[") || content.startsWith("-")) return false;
  if (
    content.startsWith("#") ||
    content.startsWith("$") ||
    content.startsWith("{$")
  )
    return false;
  if (ARR_HDR.test(content)) return false;
  if (TAB_HDR.test(content)) return false;
  const eq = findEq(content);
  const col = findColon(content);
  if (eq === -1 && col === -1) return false;
  const candidates = [eq, col].filter((i) => i !== -1);
  const sep = Math.min(...candidates);
  const keyCandidate = content.slice(0, sep).trim();
  for (const c of [",", "[", "]", "{", "}", "\t", "$"]) {
    if (keyCandidate.includes(c)) return false;
  }
  return true;
}

class Lines {
  constructor(text) {
    this.lines = text.split("\n");
    this.pos = 0;
  }
  _scan(i) {
    while (i < this.lines.length) {
      const raw = this.lines[i];
      if (raw.trim()) {
        let ind = 0;
        while (ind < raw.length && raw[ind] === "\t") ind++;
        return { i, ind, content: raw.slice(ind) };
      }
      i++;
    }
    return null;
  }
  peek() {
    const r = this._scan(this.pos);
    return r ? [r.ind, r.content] : null;
  }
  consume() {
    const r = this._scan(this.pos);
    if (!r) throw new Error("No more lines");
    this.pos = r.i + 1;
    return [r.ind, r.content];
  }
  insert(depth, content) {
    this.lines.splice(this.pos, 0, "\t".repeat(depth) + content);
  }
}

function makeDecoder(exactMap, prefixMap) {
  const sortedPrefix = [...prefixMap].sort((a, b) => b[1].length - a[1].length);

  function resolveToken(tok) {
    if (tok.startsWith("$")) {
      return exactMap.has(tok) ? { hit: true, value: exactMap.get(tok) } : null;
    }
    if (tok.startsWith("{$")) {
      const close = tok.indexOf("}");
      if (close !== -1) {
        const token = tok.slice(1, close);
        const suffix = tok.slice(close + 1);
        for (const [t, prefixVal] of sortedPrefix) {
          if (t === token) return { hit: true, value: prefixVal + suffix };
        }
      }
    }
    return null;
  }

  function decPrim(tok) {
    tok = tok.trim();
    const r = resolveToken(tok);
    if (r) return r.value;
    return decPrimBase(tok);
  }

  function decStr(tok) {
    tok = tok.trim();
    const r = resolveToken(tok);
    if (r) return r.value;
    return decStrTok(tok);
  }

  function parseValue(lines) {
    const p = lines.peek();
    if (p === null) return null;
    const [ind, content] = p;
    if (content === "{}" || content === "[]" || content === "[0]") {
      lines.consume();
      return content === "{}" ? {} : [];
    }
    if (TAB_HDR.test(content)) return parseTabular(lines);
    if (ARR_HDR.test(content)) return parseArray(lines);
    if (looksLikeObjLine(content)) return parseObject(lines, ind);
    const [, c2] = lines.consume();
    return decPrim(c2);
  }

  function parseTabular(lines) {
    const [, content] = lines.consume();
    const m = TAB_HDR.exec(content);
    const count = parseInt(m[1], 10);
    const keysStr = m[2].trim();
    let paths;
    if (keysStr.startsWith("csv:")) {
      paths = csvSplit(keysStr.slice(4)).map(decStrTok);
    } else {
      paths = keysStr.split(" ");
    }
    const hasDots = paths.some((p) => p.includes("."));
    const rows = [];
    for (let r = 0; r < count; r++) {
      const [, rowContent] = lines.consume();
      const cells = rowContent.split("\t");
      if (cells.length !== paths.length) {
        throw new Error(
          `Expected ${paths.length} cells, got ${cells.length}: ${JSON.stringify(rowContent)}`
        );
      }
      if (hasDots) {
        const obj = {};
        for (let c = 0; c < paths.length; c++)
          setPath(obj, paths[c], decPrim(cells[c]));
        rows.push(obj);
      } else {
        const obj = {};
        for (let c = 0; c < paths.length; c++) obj[paths[c]] = decPrim(cells[c]);
        rows.push(obj);
      }
    }
    return rows;
  }

  function parseArray(lines) {
    const [, content] = lines.consume();
    const m = ARR_HDR.exec(content);
    if (!m) throw new Error(`Expected array header, got: ${JSON.stringify(content)}`);
    const count = parseInt(m[1], 10);
    const rest = m[2].trim();
    if (count === 0) return [];
    if (rest) return csvSplit(rest).map(decPrim);
    const items = [];
    for (let r = 0; r < count; r++) {
      const p = lines.peek();
      if (p === null) throw new Error("Unexpected EOF in dash list");
      const [itemInd, itemContent] = p;
      if (!itemContent.startsWith("-"))
        throw new Error(`Expected '- item', got: ${JSON.stringify(itemContent)}`);
      items.push(parseDashItem(lines, itemInd));
    }
    return items;
  }

  function parseDashItem(lines, itemIndent) {
    const [ind, content] = lines.consume();
    if (content === "-") return null;
    let rest;
    if (content.startsWith("- ")) rest = content.slice(2).trim();
    else if (content.startsWith("-")) rest = content.slice(1).trim();
    else throw new Error(`Expected dash item: ${JSON.stringify(content)}`);
    if (!rest) return null;

    if (TAB_HDR.test(rest)) {
      lines.insert(ind + 1, rest);
      return parseTabular(lines);
    }
    if (ARR_HDR.test(rest)) {
      lines.insert(ind + 1, rest);
      return parseArray(lines);
    }
    if (looksLikeObjLine(rest)) {
      lines.insert(ind + 1, rest);
      return parseObject(lines, ind + 1);
    }
    return decPrim(rest);
  }

  function parseObject(lines, objIndent) {
    const obj = {};
    while (true) {
      const p = lines.peek();
      if (p === null) break;
      const [ind, content] = p;
      if (ind !== objIndent) break;
      if (!looksLikeObjLine(content)) break;

      const [, content2] = lines.consume();
      const eq = findEq(content2);
      const col = findColon(content2);

      if (eq !== -1 && (col === -1 || eq < col)) {
        const rawKey = content2.slice(0, eq).trim();
        const key = decStr(rawKey);
        const rest = content2.slice(eq + 1).trim();
        if (TAB_HDR.test(rest)) {
          lines.insert(objIndent + 1, rest);
          obj[key] = parseTabular(lines);
          continue;
        }
        if (ARR_HDR.test(rest)) {
          lines.insert(objIndent + 1, rest);
          obj[key] = parseArray(lines);
          continue;
        }
        if (rest === "{}") {
          obj[key] = {};
          continue;
        }
        obj[key] = decPrim(rest);
      } else {
        if (col === -1) break;
        const rawKey = content2.slice(0, col).trim();
        const key = decStr(rawKey);
        const rest = content2.slice(col + 1).trim();
        if (!rest) {
          const p2 = lines.peek();
          if (p2 === null) {
            obj[key] = null;
            continue;
          }
          const [childInd] = p2;
          if (childInd <= objIndent) {
            obj[key] = null;
            continue;
          }
          obj[key] = parseValue(lines);
        } else {
          if (TAB_HDR.test(rest)) {
            lines.insert(objIndent + 1, rest);
            obj[key] = parseTabular(lines);
            continue;
          }
          if (ARR_HDR.test(rest)) {
            lines.insert(objIndent + 1, rest);
            obj[key] = parseArray(lines);
            continue;
          }
          obj[key] = decPrim(rest);
        }
      }
    }
    return obj;
  }

  return { parseValue };
}

/**
 * Decode a JTF string back to a JSON-compatible value.
 * Inverse of {@link encode}: decode(encode(x)) deep-equals x.
 * @param {string} text JTF text.
 * @returns {*} The decoded value.
 */
export function decode(text) {
  const linesList = text.split("\n");
  const exactMap = new Map();
  const prefixMap = [];
  let cursor = 0;

  if (linesList.length && linesList[0].trim() === "#vdf:v2") {
    cursor += 1;
    if (cursor < linesList.length && linesList[cursor].trim() === "#dict:") {
      cursor += 1;
      while (cursor < linesList.length && linesList[cursor].trim() !== "#end") {
        const line = linesList[cursor].trim();
        cursor += 1;
        if (!line) continue;
        const pm = /^(\$\d+)~=(.+)$/.exec(line);
        if (pm) {
          prefixMap.push([pm[1], decStrTok(pm[2])]);
          continue;
        }
        const em = /^(\$\d+)=(.+)$/.exec(line);
        if (em) {
          exactMap.set(em[1], decStrTok(em[2]));
          continue;
        }
      }
      if (cursor < linesList.length && linesList[cursor].trim() === "#end")
        cursor += 1;
    }
  }

  const bodyText = linesList.slice(cursor).join("\n");
  const dec = makeDecoder(exactMap, prefixMap);
  const lines = new Lines(bodyText);
  return dec.parseValue(lines);
}

export { heuristicCost };
