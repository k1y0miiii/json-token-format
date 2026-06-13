"""JTF core: encoder and decoder.

This is a refactor of the reference implementation (``json-token-format/jtf.py``)
and is the **source of truth** for JTF behaviour. The conformance vectors are
generated from and validated against this module.

The one substantive change from the reference: the value-dictionary
break-even analysis uses a pluggable cost function that defaults to a
deterministic, dependency-free heuristic (:func:`jtf.cost.heuristic_cost`)
instead of ``tiktoken``. This makes ``encode`` output reproducible across
machines and languages — a hard requirement for a shared conformance suite.
Pass ``cost_fn=jtf.cost.tiktoken_cost`` to optimize for a real tokenizer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Optional

from .cost import CostFn, heuristic_cost

# ---------------------------------------------------------------------------
# String safety
# ---------------------------------------------------------------------------

# Characters forbidden in bare (unquoted) strings.
#   "=" is the key=value separator.
#   ":" is the key: (nested block) separator.
#   "#" / "$" / "@" / "[" / "{" / "-" have structural meaning at line start.
#   tab is the indentation and cell separator; comma separates inline values.
_SAFE_RE = re.compile(r'^[^\n\r,=:|"\\#\[\]{}\t$@]+$')
_RESERVED = frozenset({"null", "true", "false"})


def _safe(s: str) -> bool:
    """True if *s* may be written bare (unquoted) as a key or string value."""
    if not s:
        return False
    if s in _RESERVED:
        return False
    if s != s.strip():
        return False
    if s[0] in ('"', "[", "{", "-", "#", "=", "$", "@"):
        return False
    if "= " in s or s.endswith("="):
        return False
    return bool(_SAFE_RE.match(s))


def _looks_numeric(s: str) -> bool:
    """True if the bare form of *s* would decode back as a number, not a string."""
    try:
        if "." in s or "e" in s.lower():
            float(s)
        else:
            int(s)
        return True
    except ValueError:
        return False


def _es(s: str) -> str:
    """Encode a key/string token: bare when safe, otherwise JSON-quoted."""
    return s if _safe(s) else json.dumps(s, ensure_ascii=False)


def _ep(v: Any) -> str:
    """Encode a primitive value to its JTF token form."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return json.dumps(v)
    # String: quote if unsafe OR if the bare form would parse as number/bool/null.
    if _looks_numeric(v):
        return json.dumps(v, ensure_ascii=False)
    return _es(v)


def _is_prim(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


# ---------------------------------------------------------------------------
# Uniform-array detection
# ---------------------------------------------------------------------------

def _dotted_paths_ordered(obj: dict, prefix: str = "") -> Optional[list]:
    """Ordered list of dotted leaf-key paths for a (possibly nested) dict.

    Returns ``None`` if any leaf is a list (cannot be flattened into a column).
    """
    result: list = []
    for k, v in obj.items():
        path = k if not prefix else f"{prefix}.{k}"
        if _is_prim(v):
            result.append(path)
        elif isinstance(v, dict):
            child = _dotted_paths_ordered(v, path)
            if child is None:
                return None
            result.extend(child)
        else:
            return None
    return result


def _dotted_paths_set(obj: dict, prefix: str = "") -> Optional[set]:
    r = _dotted_paths_ordered(obj, prefix)
    return None if r is None else set(r)


def _get_path(obj: dict, path: str) -> Any:
    cur = obj
    for p in path.split("."):
        cur = cur[p]
    return cur


def _set_path(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _is_nested_uniform(arr: list):
    """Detect a tabular array.

    Returns ``(paths_ordered, has_dots)`` when *arr* is a non-empty list of
    objects that all expose exactly the same set of dotted leaf-paths with
    all-primitive leaves; ``None`` otherwise.
    """
    if not arr or not isinstance(arr[0], dict) or not arr[0]:
        return None
    paths0 = _dotted_paths_ordered(arr[0])
    if paths0 is None or not paths0:
        return None
    paths0_set = set(paths0)
    for item in arr[1:]:
        if not isinstance(item, dict):
            return None
        ps = _dotted_paths_set(item)
        if ps != paths0_set:
            return None
    has_dots = any("." in p for p in paths0)
    return (paths0, has_dots)


# ---------------------------------------------------------------------------
# Value dictionary (break-even analysis)
# ---------------------------------------------------------------------------

_PREFIX_MIN_LEN = 10


class DictEntry:
    __slots__ = ("token", "value", "is_prefix")

    def __init__(self, token: str, value: str, is_prefix: bool = False):
        self.token = token
        self.value = value
        self.is_prefix = is_prefix


def _collect_values(data: Any, vctr: Counter, pctr: Counter) -> None:
    if isinstance(data, str):
        vctr[data] += 1
        _note_prefix(data, pctr)
    elif isinstance(data, list):
        for x in data:
            _collect_values(x, vctr, pctr)
    elif isinstance(data, dict):
        for v in data.values():
            _collect_values(v, vctr, pctr)


def _note_prefix(s: str, pctr: Counter) -> None:
    if s.startswith("http://") or s.startswith("https://"):
        idx = s.rfind("/", 8)
        if idx > 8 and idx < len(s) - 1:
            prefix = s[: idx + 1]
            if len(prefix) >= _PREFIX_MIN_LEN:
                pctr[prefix] += 1
    if len(s) > 10 and s[10] == "T" and re.match(r"^\d{4}-\d{2}-\d{2}T", s):
        pctr[s[:11]] += 1


def _val_repr(s: str) -> str:
    """How a string value appears in the body / dict (quoted or bare)."""
    return s if _safe(s) else json.dumps(s, ensure_ascii=False)


def _dict_header_overhead(cost_fn: CostFn) -> int:
    """Fixed token cost of the dictionary header+footer block."""
    return cost_fn("#vdf:v2\n#dict:\n") + cost_fn("#end\n")


def _build_dictionary(data: Any, cost_fn: CostFn, min_net_savings: int = 2):
    """Build a profitable value dictionary.

    Returns ``(entries, value_map, prefix_covered)``. Each admitted entry must
    save more than ``min_net_savings`` tokens net of its own dict line, and the
    dictionary as a whole is discarded if its combined net savings do not cover
    the fixed header/footer overhead.
    """
    vctr: Counter = Counter()
    pctr: Counter = Counter()
    _collect_values(data, vctr, pctr)

    entries: list[DictEntry] = []
    value_map: dict[str, str] = {}
    prefix_covered: set[str] = set()
    idx = 0

    entry_savings: list[int] = []
    entry_costs: list[int] = []

    # Phase 1: exact-value entries, most valuable first (count * repr-cost).
    for val, count in sorted(
        vctr.items(), key=lambda x: -x[1] * cost_fn(_val_repr(x[0]))
    ):
        if count < 2:
            continue
        val_repr = _val_repr(val)
        val_cost = cost_fn(val_repr)
        tok_name = f"${idx}"
        tok_cost = cost_fn(tok_name)
        if val_cost <= tok_cost:
            continue
        dict_line = f"  {tok_name}={val_repr}"
        entry_cost = cost_fn(dict_line)
        gross_savings = (val_cost - tok_cost) * count
        if gross_savings > entry_cost + min_net_savings:
            entries.append(DictEntry(tok_name, val, is_prefix=False))
            value_map[val] = tok_name
            entry_savings.append(gross_savings)
            entry_costs.append(entry_cost)
            idx += 1

    # Phase 2: URL / timestamp prefix entries.
    for prefix, p_count in pctr.most_common():
        if p_count < 2:
            continue
        matching = [v for v in vctr if v.startswith(prefix) and v not in value_map]
        if len(matching) < 2:
            continue
        prefix_repr = _val_repr(prefix)
        tok_name = f"${idx}"
        tok_cost = cost_fn(tok_name)
        dict_line = f"  {tok_name}={prefix_repr}"
        entry_cost = cost_fn(dict_line)
        gross_savings = 0
        for v in matching:
            count = vctr[v]
            suffix = v[len(prefix):]
            full_cost = cost_fn(_val_repr(v))
            new_cost = cost_fn("{" + tok_name + "}" + suffix)
            gross_savings += (full_cost - new_cost) * count
        if gross_savings > entry_cost + min_net_savings:
            entries.append(DictEntry(tok_name, prefix, is_prefix=True))
            for v in matching:
                prefix_covered.add(v)
            entry_savings.append(gross_savings)
            entry_costs.append(entry_cost)
            idx += 1

    # Global check: total net savings must exceed the fixed header/footer cost.
    if entries:
        fixed_overhead = _dict_header_overhead(cost_fn)
        net = sum(entry_savings) - sum(entry_costs) - fixed_overhead
        if net <= 0:
            return [], {}, set()

    return entries, value_map, prefix_covered


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _Encoder:
    def __init__(self, value_map: dict, prefix_covered: set, prefix_tokens: list):
        self.value_map = value_map
        self.prefix_covered = prefix_covered
        # Longest prefix first for greedy matching.
        self.prefix_tokens = sorted(prefix_tokens, key=lambda x: -len(x[0]))

    def enc_str(self, s: str) -> str:
        if s in self.value_map:
            return self.value_map[s]
        if s in self.prefix_covered:
            for prefix, tok in self.prefix_tokens:
                if s.startswith(prefix):
                    return "{" + tok + "}" + s[len(prefix):]
        return _ep(s)

    def enc_val(self, v: Any) -> str:
        if isinstance(v, str):
            return self.enc_str(v)
        return _ep(v)

    def enc_obj(self, obj: dict, depth: int) -> list:
        if not obj:
            return ["{}"]
        pad = "\t" * depth
        lines: list = []
        for k, v in obj.items():
            ek = _es(k)
            if _is_prim(v):
                lines.append(pad + f"{ek}={self.enc_val(v)}")
            elif isinstance(v, list):
                arr_lines = self.enc_array(v, depth)
                if len(arr_lines) == 1:
                    lines.append(pad + f"{ek}={arr_lines[0]}")
                else:
                    lines.append(pad + f"{ek}:{arr_lines[0]}")
                    lines.extend(arr_lines[1:])
            elif isinstance(v, dict):
                if not v:
                    lines.append(pad + ek + "={}")
                else:
                    lines.append(pad + f"{ek}:")
                    lines.extend(self.enc_obj(v, depth + 1))
            else:
                raise TypeError(f"Unsupported type: {type(v)}")
        return lines

    def enc_array(self, arr: list, depth: int) -> list:
        pad = "\t" * depth
        n = len(arr)

        if n == 0:
            return ["[0]"]

        if all(_is_prim(x) for x in arr):
            inner = ",".join(self.enc_val(x) for x in arr)
            return [f"[{n}] {inner}"]

        uni = _is_nested_uniform(arr)
        if uni is not None:
            paths, _has_dots = uni
            row_pad = pad + "\t"
            safe_keys = all(_safe(p) and " " not in p for p in paths)
            if safe_keys:
                hdr = " ".join(paths)
            else:
                hdr = "csv:" + ",".join(_es(p) for p in paths)
            lines = [f"#{n} {hdr}"]
            for item in arr:
                cells = "\t".join(self.enc_val(_get_path(item, p)) for p in paths)
                lines.append(row_pad + cells)
            return lines

        # Mixed / heterogeneous: dash list.
        cont_pad = pad + "\t\t"
        lines = [f"[{n}]"]
        for item in arr:
            item_lines = self.enc_item(item)
            if item_lines:
                lines.append(pad + "\t- " + item_lines[0])
                for subsequent in item_lines[1:]:
                    lines.append(cont_pad + subsequent)
            else:
                lines.append(pad + "\t-")
        return lines

    def enc_item(self, val: Any) -> list:
        """Lines for a dash-list item, emitted at depth 0 (no leading tabs)."""
        if _is_prim(val):
            return [self.enc_val(val)]
        if isinstance(val, list):
            return self.enc_array(val, 0)
        if isinstance(val, dict):
            return self.enc_obj(val, 0)
        raise TypeError(f"Unsupported type: {type(val)}")


def encode(data: Any, cost_fn: CostFn = heuristic_cost) -> str:
    """Encode a JSON-compatible Python object to a JTF string.

    Args:
        data: Any JSON-serializable value (``dict``, ``list``, ``str``,
            ``int``, ``float``, ``bool``, ``None``).
        cost_fn: Token-cost estimator for the value-dictionary break-even
            analysis. Defaults to the deterministic
            :func:`jtf.cost.heuristic_cost`, which is what the conformance
            suite uses. Pass :func:`jtf.cost.tiktoken_cost` to optimize for a
            real GPT tokenizer.

    Returns:
        The JTF-encoded string (no trailing newline).
    """
    entries, value_map, prefix_covered = _build_dictionary(data, cost_fn)

    prefix_tokens = [(e.value, e.token) for e in entries if e.is_prefix]
    enc = _Encoder(value_map, prefix_covered, prefix_tokens)

    if isinstance(data, dict):
        body_lines = enc.enc_obj(data, 0)
    elif isinstance(data, list):
        body_lines = enc.enc_array(data, 0)
    elif _is_prim(data):
        body_lines = [enc.enc_val(data)]
    else:
        raise TypeError(f"Unsupported top-level type: {type(data)}")

    if not entries:
        return "\n".join(body_lines)

    header_lines = ["#vdf:v2", "#dict:"]
    for e in entries:
        val_repr = _val_repr(e.value)
        sep = "~=" if e.is_prefix else "="
        header_lines.append(f"  {e.token}{sep}{val_repr}")
    header_lines.append("#end")

    return "\n".join(header_lines) + "\n" + "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _Lines:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0

    def eof(self) -> bool:
        return self.pos >= len(self.lines)

    def peek(self):
        i = self.pos
        while i < len(self.lines):
            raw = self.lines[i]
            if raw.strip():
                ind = 0
                while ind < len(raw) and raw[ind] == "\t":
                    ind += 1
                return (ind, raw[ind:])
            i += 1
        return None

    def consume(self):
        while self.pos < len(self.lines):
            raw = self.lines[self.pos]
            self.pos += 1
            if raw.strip():
                ind = 0
                while ind < len(raw) and raw[ind] == "\t":
                    ind += 1
                return (ind, raw[ind:])
        raise EOFError("No more lines")

    def insert(self, depth: int, content: str):
        self.lines.insert(self.pos, "\t" * depth + content)


_TAB_HDR = re.compile(r"^#(\d+) (.+)$")
_ARR_HDR = re.compile(r"^\[(\d+)\](.*)$")


def _csv_split(s: str) -> list:
    """Split on commas, respecting double-quoted fields."""
    fields, cur, in_q = [], [], False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"' and not in_q:
            in_q = True
            cur.append(c)
        elif c == '"' and in_q:
            if i + 1 < len(s) and s[i + 1] == '"':
                cur.append('"')
                i += 1
            else:
                in_q = False
                cur.append(c)
        elif c == "," and not in_q:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    fields.append("".join(cur))
    return fields


def _dec_str_tok(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith('"'):
        return json.loads(tok)
    return tok


def _dec_prim_base(tok: str) -> Any:
    """Decode a bare token to a Python primitive (no dictionary resolution)."""
    tok = tok.strip()
    if tok == "null":
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    if tok == "{}":
        return {}
    if tok == "[]":
        return []
    if tok.startswith('"'):
        return json.loads(tok)
    try:
        if "." in tok or "e" in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        pass
    return tok


def _find_eq(s: str) -> int:
    """First unquoted '=' (the key=value separator)."""
    in_q = False
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i - 1] != "\\"):
            in_q = not in_q
        if not in_q and c == "=":
            return i
    return -1


def _find_colon(s: str) -> int:
    """First unquoted ':' (the key: nested-block separator)."""
    in_q = False
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i - 1] != "\\"):
            in_q = not in_q
        if not in_q and c == ":":
            return i
    return -1


def _looks_like_obj_line(content: str) -> bool:
    if content.startswith("[") or content.startswith("-"):
        return False
    if content.startswith("#") or content.startswith("$") or content.startswith("{$"):
        return False
    if _ARR_HDR.match(content):
        return False
    if _TAB_HDR.match(content):
        return False
    eq = _find_eq(content)
    col = _find_colon(content)
    if eq == -1 and col == -1:
        return False
    sep = min(i for i in [eq, col] if i != -1)
    key_candidate = content[:sep].strip()
    if any(c in key_candidate for c in [",", "[", "]", "{", "}", "\t", "$"]):
        return False
    return True


class _Decoder:
    def __init__(self, exact_map: dict, prefix_map: list):
        self.exact_map = exact_map
        # Longest prefix-value first.
        self.prefix_map = sorted(prefix_map, key=lambda x: -len(x[1]))

    def dec_prim(self, tok: str) -> Any:
        tok = tok.strip()
        if tok.startswith("$"):
            return self.exact_map.get(tok, tok)
        if tok.startswith("{$"):
            close = tok.find("}")
            if close != -1:
                token = tok[1:close]
                suffix = tok[close + 1:]
                for t, prefix_val in self.prefix_map:
                    if t == token:
                        return prefix_val + suffix
        return _dec_prim_base(tok)

    def dec_str(self, tok: str) -> str:
        tok = tok.strip()
        if tok.startswith("$"):
            return self.exact_map.get(tok, tok)
        if tok.startswith("{$"):
            close = tok.find("}")
            if close != -1:
                token = tok[1:close]
                suffix = tok[close + 1:]
                for t, prefix_val in self.prefix_map:
                    if t == token:
                        return prefix_val + suffix
        return _dec_str_tok(tok)

    def parse_value(self, lines: _Lines, min_indent: int) -> Any:
        p = lines.peek()
        if p is None:
            return None
        ind, content = p

        if content in ("{}", "[]", "[0]"):
            lines.consume()
            if content == "{}":
                return {}
            return []

        if _TAB_HDR.match(content):
            return self.parse_tabular(lines, ind)

        if _ARR_HDR.match(content):
            return self.parse_array(lines, ind)

        if _looks_like_obj_line(content):
            return self.parse_object(lines, ind)

        _, content2 = lines.consume()
        return self.dec_prim(content2)

    def parse_tabular(self, lines: _Lines, arr_indent: int) -> list:
        _ind, content = lines.consume()
        m = _TAB_HDR.match(content)
        count = int(m.group(1))
        keys_str = m.group(2).strip()

        if keys_str.startswith("csv:"):
            paths = [_dec_str_tok(f) for f in _csv_split(keys_str[4:])]
        else:
            paths = keys_str.split(" ")

        has_dots = any("." in p for p in paths)

        rows = []
        for _ in range(count):
            _, row_content = lines.consume()
            cells = row_content.split("\t")
            if len(cells) != len(paths):
                raise ValueError(
                    f"Expected {len(paths)} cells, got {len(cells)}: {row_content!r}"
                )
            if has_dots:
                obj: dict = {}
                for path, cell in zip(paths, cells):
                    _set_path(obj, path, self.dec_prim(cell))
                rows.append(obj)
            else:
                rows.append(
                    {p: self.dec_prim(c) for p, c in zip(paths, cells)}
                )
        return rows

    def parse_array(self, lines: _Lines, arr_indent: int) -> list:
        _ind, content = lines.consume()
        m = _ARR_HDR.match(content)
        if not m:
            raise ValueError(f"Expected array header, got: {content!r}")
        count = int(m.group(1))
        rest = m.group(2).strip()

        if count == 0:
            return []

        if rest:
            return [self.dec_prim(x) for x in _csv_split(rest)]

        items = []
        for _ in range(count):
            p = lines.peek()
            if p is None:
                raise ValueError("Unexpected EOF in dash list")
            _item_ind, item_content = p
            if not item_content.startswith("-"):
                raise ValueError(f"Expected '- item', got: {item_content!r}")
            items.append(self.parse_dash_item(lines, _item_ind))
        return items

    def parse_dash_item(self, lines: _Lines, item_indent: int) -> Any:
        ind, content = lines.consume()
        if content == "-":
            return None
        if content.startswith("- "):
            rest = content[2:].strip()
        elif content.startswith("-"):
            rest = content[1:].strip()
        else:
            raise ValueError(f"Expected dash item: {content!r}")
        if not rest:
            return None

        if _TAB_HDR.match(rest):
            lines.insert(ind + 1, rest)
            return self.parse_tabular(lines, ind + 1)

        if _ARR_HDR.match(rest):
            lines.insert(ind + 1, rest)
            return self.parse_array(lines, ind + 1)

        if _looks_like_obj_line(rest):
            lines.insert(ind + 1, rest)
            return self.parse_object(lines, ind + 1)

        return self.dec_prim(rest)

    def parse_object(self, lines: _Lines, obj_indent: int) -> dict:
        obj: dict = {}
        while True:
            p = lines.peek()
            if p is None:
                break
            ind, content = p
            if ind != obj_indent:
                break
            if not _looks_like_obj_line(content):
                break

            _, content2 = lines.consume()
            eq = _find_eq(content2)
            col = _find_colon(content2)

            if eq != -1 and (col == -1 or eq < col):
                # key=value
                raw_key = content2[:eq].strip()
                key = self.dec_str(raw_key)
                rest = content2[eq + 1:].strip()

                if _TAB_HDR.match(rest):
                    lines.insert(obj_indent + 1, rest)
                    obj[key] = self.parse_tabular(lines, obj_indent + 1)
                    continue
                if _ARR_HDR.match(rest):
                    lines.insert(obj_indent + 1, rest)
                    obj[key] = self.parse_array(lines, obj_indent + 1)
                    continue
                if rest == "{}":
                    obj[key] = {}
                    continue
                obj[key] = self.dec_prim(rest)
            else:
                # key: (nested block follows)
                if col == -1:
                    break
                raw_key = content2[:col].strip()
                key = self.dec_str(raw_key)
                rest = content2[col + 1:].strip()

                if not rest:
                    p2 = lines.peek()
                    if p2 is None:
                        obj[key] = None
                        continue
                    child_ind, _ = p2
                    if child_ind <= obj_indent:
                        obj[key] = None
                        continue
                    obj[key] = self.parse_value(lines, child_ind)
                else:
                    if _TAB_HDR.match(rest):
                        lines.insert(obj_indent + 1, rest)
                        obj[key] = self.parse_tabular(lines, obj_indent + 1)
                        continue
                    if _ARR_HDR.match(rest):
                        lines.insert(obj_indent + 1, rest)
                        obj[key] = self.parse_array(lines, obj_indent + 1)
                        continue
                    obj[key] = self.dec_prim(rest)

        return obj


def decode(text: str) -> Any:
    """Decode a JTF string back to a JSON-compatible Python object.

    Inverse of :func:`encode`: ``decode(encode(x)) == x`` for any
    JSON-serializable ``x``.
    """
    lines_list = text.splitlines()
    exact_map: dict[str, str] = {}
    prefix_map: list = []
    cursor = 0

    if lines_list and lines_list[0].strip() == "#vdf:v2":
        cursor += 1
        if cursor < len(lines_list) and lines_list[cursor].strip() == "#dict:":
            cursor += 1
            while cursor < len(lines_list) and lines_list[cursor].strip() != "#end":
                line = lines_list[cursor].strip()
                cursor += 1
                if not line:
                    continue
                pm = re.match(r"^(\$\d+)~=(.+)$", line)
                if pm:
                    prefix_map.append((pm.group(1), _dec_str_tok(pm.group(2))))
                    continue
                em = re.match(r"^(\$\d+)=(.+)$", line)
                if em:
                    exact_map[em.group(1)] = _dec_str_tok(em.group(2))
                    continue
            if cursor < len(lines_list) and lines_list[cursor].strip() == "#end":
                cursor += 1

    body_text = "\n".join(lines_list[cursor:])
    dec = _Decoder(exact_map, prefix_map)
    ln = _Lines(body_text)
    return dec.parse_value(ln, 0)
