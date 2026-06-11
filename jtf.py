#!/usr/bin/env python3
"""
JTF v2 - JSON Token Format (consolidated)
==========================================
A lossless, round-trippable JSON encoding that minimises tiktoken token count
for LLM prompts.

Format spec (v2):

  Primitives:
    null, true, false written literally.
    Numbers in canonical JSON form.
    Strings: unquoted when safe (no = : | tab newline leading/trailing space
             no collision with null/true/false, no starting with " [ { - # $ =).
    Unsafe strings are double-quoted with JSON escape sequences.
    Numeric-looking strings (would be mis-parsed as a number/bool/null) are
    always double-quoted even when they pass the character safety check.

  Objects (tab-indented per level):
    key=value         <- primitive value  (saves 1 token vs "key: value")
    key:              <- child block follows on next indented lines
      child=...

  Empty containers:
    {}  []  [0]

  Primitive arrays (inline comma-separated):
    [N] v1,v2,v3

  Tabular arrays (flat-uniform objects — same keys, all-primitive values):
    #N k1 k2 k3         <- header: N count, space-separated key names
    \tv1\tv2\tv3        <- one tab-indented row per item, cells tab-separated
    ...
    Keys containing spaces or unsafe characters use the csv: prefix form:
    #N csv:"k 1","k 2"
    (rows still tab-separated)

  Nested-tabular arrays (nested-uniform — uniform objects with nested dicts,
  all leaf values primitive):
    #N.dotted path1 path2.sub    <- dotted-path header
    \tv1\tv2                     <- tab-indented, tab-separated rows
    Reconstruction: dotted paths are split on "." to rebuild nested dicts.

  Mixed/heterogeneous arrays (dash list):
    [N]
    \t- item1
    \t- key=val
    ...

  Value dictionary (applied automatically when profitable):
    #vdf:v2
    #dict:
      $0=frequently_repeated_string
      $1~=https://common.prefix/     <- ~= marks a URL prefix alias
    #end
    (body follows; any string value encoded via dictionary uses $N or {$N}suffix)

  Top-level value is written at indent 0 with no enclosing braces.

Grammar sketch:

  document  ::= [header] body
  header    ::= "#vdf:v2\n#dict:\n" entry* "#end\n"
  entry     ::= "  " token ("=" | "~=") repr "\n"
  token     ::= "$" digits
  body      ::= value
  value     ::= primitive | object | array
  object    ::= (kv_line | nested_line)*
  kv_line   ::= key "=" prim_val
  nested_line ::= key ":" "\n" INDENT value DEDENT
  array     ::= "[0]" | "[N] " inline_csv
              | "#N " hdr "\n" (INDENT row "\n")*
              | "[N]" "\n" (INDENT "- " item "\n")*
  INDENT / DEDENT via leading tabs (1 tab per level)
"""

import json
import sys
import re
import argparse
import math
from collections import Counter

# ---------------------------------------------------------------------------
# String safety
# ---------------------------------------------------------------------------

# Characters forbidden in bare (unquoted) strings.
# "=" is forbidden because it is the key=value separator.
# "#" at start is the array/dict header marker.
# "$" at start is a value-dictionary token.
_SAFE_RE = re.compile(r'^[^\n\r,=:|"\\#\[\]{}\t$@]+$')
_RESERVED = frozenset({'null', 'true', 'false'})


def _safe(s: str) -> bool:
    if not s:
        return False
    if s in _RESERVED:
        return False
    if s != s.strip():
        return False
    if s[0] in ('"', '[', '{', '-', '#', '=', '$', '@'):
        return False
    if '= ' in s or s.endswith('='):
        return False
    return bool(_SAFE_RE.match(s))


def _looks_numeric(s: str) -> bool:
    """True if s would be decoded as int/float by _dec_prim (numeric-string quoting fix)."""
    try:
        if '.' in s or 'e' in s.lower():
            float(s)
        else:
            int(s)
        return True
    except ValueError:
        return False


def _es(s: str) -> str:
    """Encode a string (key or value); quote if not safe."""
    return s if _safe(s) else json.dumps(s, ensure_ascii=False)


def _ep(v) -> str:
    """Encode a primitive value."""
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return json.dumps(v)
    # String: quote if unsafe OR if the bare form would parse as a number/bool/null.
    if _looks_numeric(v):
        return json.dumps(v, ensure_ascii=False)
    return _es(v)


def _is_prim(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


# ---------------------------------------------------------------------------
# Uniform-array detection
# ---------------------------------------------------------------------------

def _dotted_paths_ordered(obj: dict, prefix: str = '') -> list | None:
    """Return ordered list of dotted leaf-key paths for a (possibly nested) dict.
    Returns None if any leaf is a list (can't flatten)."""
    result = []
    for k, v in obj.items():
        path = k if not prefix else f'{prefix}.{k}'
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


def _dotted_paths_set(obj: dict, prefix: str = '') -> set | None:
    r = _dotted_paths_ordered(obj, prefix)
    return None if r is None else set(r)


def _get_path(obj: dict, path: str):
    parts = path.split('.')
    cur = obj
    for p in parts:
        cur = cur[p]
    return cur


def _set_path(obj: dict, path: str, value):
    parts = path.split('.')
    cur = obj
    for p in parts[:-1]:
        if p not in cur:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _is_nested_uniform(arr: list):
    """Check if arr is a non-empty list of nested-uniform dicts.
    Returns (paths_ordered, has_dots) or None if not uniform."""
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
    has_dots = any('.' in p for p in paths0)
    return (paths0, has_dots)


# ---------------------------------------------------------------------------
# Value dictionary (break-even analysis)
# ---------------------------------------------------------------------------

try:
    import tiktoken as _tiktoken
    _enc_cl100k = _tiktoken.get_encoding('cl100k_base')
    def _tok(s: str) -> int:
        return len(_enc_cl100k.encode(s))
    _TIKTOKEN_OK = True
except Exception:
    def _tok(s: str) -> int:
        return max(1, len(s) // 4)
    _TIKTOKEN_OK = False


_PREFIX_MIN_LEN = 10

class DictEntry:
    __slots__ = ('token', 'value', 'is_prefix')
    def __init__(self, token: str, value: str, is_prefix: bool = False):
        self.token = token
        self.value = value
        self.is_prefix = is_prefix


def _collect_values(data, vctr: Counter, pctr: Counter):
    if isinstance(data, str):
        vctr[data] += 1
        _note_prefix(data, pctr)
    elif isinstance(data, list):
        for x in data:
            _collect_values(x, vctr, pctr)
    elif isinstance(data, dict):
        for v in data.values():
            _collect_values(v, vctr, pctr)


def _note_prefix(s: str, pctr: Counter):
    if s.startswith('http://') or s.startswith('https://'):
        idx = s.rfind('/', 8)
        if idx > 8 and idx < len(s) - 1:
            prefix = s[:idx + 1]
            if len(prefix) >= _PREFIX_MIN_LEN:
                pctr[prefix] += 1
    t_idx = s.find('T')
    if t_idx == 10 and re.match(r'^\d{4}-\d{2}-\d{2}T', s):
        pctr[s[:11]] += 1


def _val_repr(s: str) -> str:
    """How the string appears in the JTF body (quoted or bare)."""
    return s if _safe(s) else json.dumps(s, ensure_ascii=False)


_DICT_HEADER_OVERHEAD = None  # computed lazily

def _dict_header_overhead(cost_fn) -> int:
    """Fixed token cost of the dictionary header+footer block (not per-entry)."""
    # '#vdf:v2\n#dict:\n' + '#end\n' — paid once when ANY dict is emitted.
    return cost_fn('#vdf:v2\n#dict:\n') + cost_fn('#end\n')


def _build_dictionary(data, cost_fn=_tok, min_net_savings: int = 2):
    """Build a profitable value dictionary.

    min_net_savings: per-entry guard — require that the net savings from an entry
    (total savings - entry line cost) exceed this threshold before admitting it.

    After all per-entry decisions are made, a second global check discards the
    entire dictionary if the combined net savings do not cover the fixed
    header/footer overhead (12 tokens for '#vdf:v2', '#dict:', '#end').
    This prevents regressions on files where only a small number of cheap
    substitutions are possible (e.g. api_response.json with 3 timestamps).

    Returns: (entries, value_map, prefix_covered)
    """
    vctr: Counter = Counter()
    pctr: Counter = Counter()
    _collect_values(data, vctr, pctr)

    entries: list[DictEntry] = []
    value_map: dict[str, str] = {}
    prefix_covered: set[str] = set()
    idx = 0

    # Compute per-entry savings tracking for the global check
    entry_savings: list[int] = []   # gross savings per entry (before entry line cost)
    entry_costs: list[int] = []     # cost of that entry's dict line

    # Phase 1: exact-value entries
    for val, count in sorted(vctr.items(), key=lambda x: -x[1] * _tok(_val_repr(x[0]))):
        if count < 2:
            continue
        val_repr = _val_repr(val)
        val_cost = cost_fn(val_repr)
        tok_name = f'${idx}'
        tok_cost = cost_fn(tok_name)
        if val_cost <= tok_cost:
            continue
        dict_line = f'  {tok_name}={val_repr}'
        entry_cost = cost_fn(dict_line)
        savings_per = val_cost - tok_cost
        gross_savings = savings_per * count
        # Per-entry break-even: gross_savings must exceed entry_cost + min_net_savings
        if gross_savings > entry_cost + min_net_savings:
            entries.append(DictEntry(tok_name, val, is_prefix=False))
            value_map[val] = tok_name
            entry_savings.append(gross_savings)
            entry_costs.append(entry_cost)
            idx += 1

    # Phase 2: prefix entries
    for prefix, p_count in pctr.most_common():
        if p_count < 2:
            continue
        matching = [v for v in vctr if v.startswith(prefix) and v not in value_map]
        if len(matching) < 2:
            continue
        prefix_repr = _val_repr(prefix)
        tok_name = f'${idx}'
        tok_cost = cost_fn(tok_name)
        dict_line = f'  {tok_name}={prefix_repr}'
        entry_cost = cost_fn(dict_line)
        gross_savings = 0
        for v in matching:
            count = vctr[v]
            full_repr = _val_repr(v)
            suffix = v[len(prefix):]
            full_cost = cost_fn(full_repr)
            new_cost = cost_fn('{' + tok_name + '}' + suffix)
            gross_savings += (full_cost - new_cost) * count
        if gross_savings > entry_cost + min_net_savings:
            entries.append(DictEntry(tok_name, prefix, is_prefix=True))
            for v in matching:
                prefix_covered.add(v)
            entry_savings.append(gross_savings)
            entry_costs.append(entry_cost)
            idx += 1

    # Global check: total net savings must exceed the fixed header/footer overhead.
    # If not, the dictionary hurts more than it helps; discard it entirely.
    if entries:
        fixed_overhead = _dict_header_overhead(cost_fn)
        total_gross = sum(entry_savings)
        total_entry_cost = sum(entry_costs)
        net = total_gross - total_entry_cost - fixed_overhead
        if net <= 0:
            return [], {}, set()

    return entries, value_map, prefix_covered


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class _Encoder:
    def __init__(self, value_map: dict, prefix_covered: set,
                 prefix_tokens: list):
        self.value_map = value_map
        self.prefix_covered = prefix_covered
        # sorted longest-first for greedy prefix matching
        self.prefix_tokens = sorted(prefix_tokens, key=lambda x: -len(x[0]))

    def enc_str(self, s: str) -> str:
        if s in self.value_map:
            return self.value_map[s]
        if s in self.prefix_covered:
            for prefix, tok in self.prefix_tokens:
                if s.startswith(prefix):
                    suffix = s[len(prefix):]
                    return '{' + tok + '}' + suffix
        return _ep(s)

    def enc_val(self, v) -> str:
        if isinstance(v, str):
            return self.enc_str(v)
        return _ep(v)

    def enc_obj(self, obj: dict, depth: int) -> list[str]:
        if not obj:
            return ['{}']
        pad = '\t' * depth
        lines = []
        for k, v in obj.items():
            ek = _es(k)
            if _is_prim(v):
                lines.append(pad + f'{ek}={self.enc_val(v)}')
            elif isinstance(v, list):
                arr_lines = self.enc_array(v, depth)
                if len(arr_lines) == 1:
                    lines.append(pad + f'{ek}={arr_lines[0]}')
                else:
                    lines.append(pad + f'{ek}:{arr_lines[0]}')
                    lines.extend(arr_lines[1:])
            elif isinstance(v, dict):
                if not v:
                    lines.append(pad + ek + '={}')
                else:
                    lines.append(pad + f'{ek}:')
                    lines.extend(self.enc_obj(v, depth + 1))
            else:
                raise TypeError(f'Unsupported type: {type(v)}')
        return lines

    def enc_array(self, arr: list, depth: int) -> list[str]:
        pad = '\t' * depth
        n = len(arr)

        if n == 0:
            return ['[0]']

        # All primitives: inline comma-separated
        if all(_is_prim(x) for x in arr):
            inner = ','.join(self.enc_val(x) for x in arr)
            return [f'[{n}] {inner}']

        # Nested-uniform tabular (covers both flat and nested objects)
        uni = _is_nested_uniform(arr)
        if uni is not None:
            paths, has_dots = uni
            row_pad = pad + '\t'
            # Build header: space-separated paths; if any path is unsafe or
            # contains spaces, fall back to csv: prefix form
            safe_keys = all(_safe(p) and ' ' not in p for p in paths)
            if safe_keys:
                hdr = ' '.join(paths)
            else:
                hdr = 'csv:' + ','.join(_es(p) for p in paths)
            lines = [f'#{n} {hdr}']
            for item in arr:
                cells = '\t'.join(self.enc_val(_get_path(item, p)) for p in paths)
                lines.append(row_pad + cells)
            return lines

        # Mixed/heterogeneous: dash list
        # Item content is at depth d+2 so the decoder sees it as a child.
        # enc_item returns lines at depth=0 (no leading tabs).
        # First line: pad + '\t- ' + line0
        # Subsequent: pad + '\t\t' + lineN  (depth+2 tabs total)
        cont_pad = pad + '\t\t'
        lines = [f'[{n}]']
        for item in arr:
            item_lines = self.enc_item(item)
            if item_lines:
                lines.append(pad + '\t- ' + item_lines[0])
                for subsequent in item_lines[1:]:
                    lines.append(cont_pad + subsequent)
            else:
                lines.append(pad + '\t-')
        return lines

    def enc_item(self, val) -> list[str]:
        """Return lines for a dash-list item at depth=0 (no leading tabs)."""
        if _is_prim(val):
            return [self.enc_val(val)]
        if isinstance(val, list):
            return self.enc_array(val, 0)
        if isinstance(val, dict):
            return self.enc_obj(val, 0)
        raise TypeError(f'Unsupported type: {type(val)}')


def encode(data) -> str:
    """Encode a Python JSON-compatible object to JTF v2 string."""
    entries, value_map, prefix_covered = _build_dictionary(data)

    prefix_tokens = [(e.value, e.token) for e in entries if e.is_prefix]
    enc = _Encoder(value_map, prefix_covered, prefix_tokens)

    if isinstance(data, dict):
        body_lines = enc.enc_obj(data, 0)
    elif isinstance(data, list):
        body_lines = enc.enc_array(data, 0)
    elif _is_prim(data):
        body_lines = [enc.enc_val(data)]
    else:
        raise TypeError(f'Unsupported top-level type: {type(data)}')

    if not entries:
        return '\n'.join(body_lines)

    header_lines = ['#vdf:v2', '#dict:']
    for e in entries:
        val_repr = _val_repr(e.value)
        sep = '~=' if e.is_prefix else '='
        header_lines.append(f'  {e.token}{sep}{val_repr}')
    header_lines.append('#end')

    return '\n'.join(header_lines) + '\n' + '\n'.join(body_lines)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class _Lines:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.pos = 0

    def eof(self) -> bool:
        return self.pos >= len(self.lines)

    def peek(self) -> tuple[int, str] | None:
        i = self.pos
        while i < len(self.lines):
            raw = self.lines[i]
            if raw.strip():
                ind = 0
                while ind < len(raw) and raw[ind] == '\t':
                    ind += 1
                return (ind, raw[ind:])
            i += 1
        return None

    def consume(self) -> tuple[int, str]:
        while self.pos < len(self.lines):
            raw = self.lines[self.pos]
            self.pos += 1
            if raw.strip():
                ind = 0
                while ind < len(raw) and raw[ind] == '\t':
                    ind += 1
                return (ind, raw[ind:])
        raise EOFError('No more lines')

    def insert(self, depth: int, content: str):
        self.lines.insert(self.pos, '\t' * depth + content)


_TAB_HDR = re.compile(r'^#(\d+) (.+)$')
_ARR_HDR = re.compile(r'^\[(\d+)\](.*)$')


def _csv_split(s: str) -> list[str]:
    """Split on commas, respecting double-quoted fields."""
    fields, cur, in_q = [], [], False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"' and not in_q:
            in_q = True; cur.append(c)
        elif c == '"' and in_q:
            if i + 1 < len(s) and s[i+1] == '"':
                cur.append('"'); i += 1
            else:
                in_q = False; cur.append(c)
        elif c == ',' and not in_q:
            fields.append(''.join(cur)); cur = []
        else:
            cur.append(c)
        i += 1
    fields.append(''.join(cur))
    return fields


def _dec_str_tok(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith('"'):
        return json.loads(tok)
    return tok


def _dec_prim_base(tok: str):
    """Decode a bare token to a Python primitive (no VDF resolution)."""
    tok = tok.strip()
    if tok == 'null':  return None
    if tok == 'true':  return True
    if tok == 'false': return False
    if tok == '{}':    return {}
    if tok == '[]':    return []
    if tok.startswith('"'):
        return json.loads(tok)
    try:
        if '.' in tok or 'e' in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        pass
    return tok


def _find_eq(s: str) -> int:
    """Find first unquoted '=' that is a key=value separator."""
    in_q = False
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            in_q = not in_q
        if not in_q and c == '=':
            return i
    return -1


def _find_colon(s: str) -> int:
    """Find first unquoted ':' for key: (nested) separator."""
    in_q = False
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            in_q = not in_q
        if not in_q and c == ':':
            return i
    return -1


def _looks_like_obj_line(content: str) -> bool:
    if content.startswith('[') or content.startswith('-'):
        return False
    if content.startswith('#') or content.startswith('$') or content.startswith('{$'):
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
    if any(c in key_candidate for c in [',', '[', ']', '{', '}', '\t', '$']):
        return False
    return True


class _Decoder:
    def __init__(self, exact_map: dict, prefix_map: list):
        self.exact_map = exact_map
        # prefix_map: [(token, prefix_value)] sorted longest-prefix-value first
        self.prefix_map = sorted(prefix_map, key=lambda x: -len(x[1]))

    def dec_prim(self, tok: str):
        tok = tok.strip()
        if tok.startswith('$'):
            return self.exact_map.get(tok, tok)
        if tok.startswith('{$'):
            close = tok.find('}')
            if close != -1:
                token = tok[1:close]
                suffix = tok[close+1:]
                for t, prefix_val in self.prefix_map:
                    if t == token:
                        return prefix_val + suffix
        return _dec_prim_base(tok)

    def dec_str(self, tok: str) -> str:
        tok = tok.strip()
        if tok.startswith('$'):
            return self.exact_map.get(tok, tok)
        if tok.startswith('{$'):
            close = tok.find('}')
            if close != -1:
                token = tok[1:close]
                suffix = tok[close+1:]
                for t, prefix_val in self.prefix_map:
                    if t == token:
                        return prefix_val + suffix
        return _dec_str_tok(tok)

    def parse_value(self, lines: _Lines, min_indent: int):
        p = lines.peek()
        if p is None:
            return None
        ind, content = p

        if content in ('{}', '[]', '[0]'):
            lines.consume()
            if content == '{}': return {}
            return []

        m = _TAB_HDR.match(content)
        if m:
            return self.parse_tabular(lines, ind)

        m2 = _ARR_HDR.match(content)
        if m2:
            return self.parse_array(lines, ind)

        if _looks_like_obj_line(content):
            return self.parse_object(lines, ind)

        _, content2 = lines.consume()
        return self.dec_prim(content2)

    def parse_tabular(self, lines: _Lines, arr_indent: int) -> list:
        ind, content = lines.consume()
        m = _TAB_HDR.match(content)
        count = int(m.group(1))
        keys_str = m.group(2).strip()

        # Parse header keys
        if keys_str.startswith('csv:'):
            paths = [_dec_str_tok(f) for f in _csv_split(keys_str[4:])]
        else:
            paths = keys_str.split(' ')

        has_dots = any('.' in p for p in paths)

        rows = []
        for _ in range(count):
            _, row_content = lines.consume()
            cells = row_content.split('\t')
            if len(cells) != len(paths):
                raise ValueError(
                    f'Expected {len(paths)} cells, got {len(cells)}: {row_content!r}')
            if has_dots:
                obj = {}
                for path, cell in zip(paths, cells):
                    _set_path(obj, path, self.dec_prim(cell))
                rows.append(obj)
            else:
                rows.append({p: self.dec_prim(c) for p, c in zip(paths, cells)})
        return rows

    def parse_array(self, lines: _Lines, arr_indent: int) -> list:
        ind, content = lines.consume()
        m = _ARR_HDR.match(content)
        if not m:
            raise ValueError(f'Expected array header, got: {content!r}')
        count = int(m.group(1))
        rest = m.group(2).strip()

        if count == 0:
            return []

        # Inline primitive: [N] v1,v2,...
        if rest:
            return [self.dec_prim(x) for x in _csv_split(rest)]

        # Dash list
        items = []
        for _ in range(count):
            p = lines.peek()
            if p is None:
                raise ValueError('Unexpected EOF in dash list')
            item_ind, item_content = p
            if not item_content.startswith('-'):
                raise ValueError(f"Expected '- item', got: {item_content!r}")
            items.append(self.parse_dash_item(lines, item_ind))
        return items

    def parse_dash_item(self, lines: _Lines, item_indent: int):
        ind, content = lines.consume()
        if content == '-':
            return None
        if content.startswith('- '):
            rest = content[2:].strip()
        elif content.startswith('-'):
            rest = content[1:].strip()
        else:
            raise ValueError(f'Expected dash item: {content!r}')
        if not rest:
            return None

        mt = _TAB_HDR.match(rest)
        if mt:
            lines.insert(ind + 1, rest)
            return self.parse_tabular(lines, ind + 1)

        m = _ARR_HDR.match(rest)
        if m:
            lines.insert(ind + 1, rest)
            return self.parse_array(lines, ind + 1)

        if _looks_like_obj_line(rest):
            lines.insert(ind + 1, rest)
            return self.parse_object(lines, ind + 1)

        return self.dec_prim(rest)

    def parse_object(self, lines: _Lines, obj_indent: int) -> dict:
        obj = {}
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
                rest = content2[eq+1:].strip()

                mt = _TAB_HDR.match(rest)
                if mt:
                    lines.insert(obj_indent + 1, rest)
                    obj[key] = self.parse_tabular(lines, obj_indent + 1)
                    continue

                m = _ARR_HDR.match(rest)
                if m:
                    lines.insert(obj_indent + 1, rest)
                    obj[key] = self.parse_array(lines, obj_indent + 1)
                    continue

                if rest == '{}':
                    obj[key] = {}
                    continue

                obj[key] = self.dec_prim(rest)

            else:
                # key: (nested block follows)
                if col == -1:
                    break
                raw_key = content2[:col].strip()
                key = self.dec_str(raw_key)
                rest = content2[col+1:].strip()

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
                    mt = _TAB_HDR.match(rest)
                    if mt:
                        lines.insert(obj_indent + 1, rest)
                        obj[key] = self.parse_tabular(lines, obj_indent + 1)
                        continue

                    m = _ARR_HDR.match(rest)
                    if m:
                        lines.insert(obj_indent + 1, rest)
                        obj[key] = self.parse_array(lines, obj_indent + 1)
                        continue

                    obj[key] = self.dec_prim(rest)

        return obj


def decode(text: str):
    """Decode a JTF v2 string to Python object."""
    lines_list = text.splitlines()
    exact_map: dict[str, str] = {}
    prefix_map: list[tuple[str, str]] = []  # (token, prefix_value)
    cursor = 0

    if lines_list and lines_list[0].strip() == '#vdf:v2':
        cursor += 1
        if cursor < len(lines_list) and lines_list[cursor].strip() == '#dict:':
            cursor += 1
            while cursor < len(lines_list) and lines_list[cursor].strip() != '#end':
                line = lines_list[cursor].strip()
                cursor += 1
                if not line:
                    continue
                pm = re.match(r'^(\$\d+)~=(.+)$', line)
                if pm:
                    token = pm.group(1)
                    val = _dec_str_tok(pm.group(2))
                    prefix_map.append((token, val))
                    continue
                em = re.match(r'^(\$\d+)=(.+)$', line)
                if em:
                    token = em.group(1)
                    val = _dec_str_tok(em.group(2))
                    exact_map[token] = val
                    continue
            if cursor < len(lines_list) and lines_list[cursor].strip() == '#end':
                cursor += 1

    body_text = '\n'.join(lines_list[cursor:])
    dec = _Decoder(exact_map, prefix_map)
    ln = _Lines(body_text)
    return dec.parse_value(ln, 0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_encode(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = encode(data)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result + '\n')
        print(f'Encoded -> {args.output}')
    else:
        print(result)


def cmd_decode(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    data = decode(text)
    result = json.dumps(data, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result + '\n')
        print(f'Decoded -> {args.output}')
    else:
        print(result)


def cmd_bench(args):
    import importlib.util as _iu, pathlib as _pl
    _jtf_v1_path = _pl.Path(__file__).resolve().parent / 'jtf_v1.py'
    _spec = _iu.spec_from_file_location('jtf_v1', _jtf_v1_path)
    _jtf_v1 = _iu.module_from_spec(_spec)
    _spec.loader.exec_module(_jtf_v1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    json_compact = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    jtf_v1_text  = _jtf_v1.encode(data)
    jtf_v2_text  = encode(data)

    if _TIKTOKEN_OK:
        def count(s): return len(_enc_cl100k.encode(s))
        method = 'tiktoken cl100k_base'
    else:
        def count(s): return max(1, len(s) // 4)
        method = 'heuristic (~4 chars/token)'

    tc  = count(json_compact)
    tv1 = count(jtf_v1_text)
    tv2 = count(jtf_v2_text)

    def pct(a, b): return (1 - a/b)*100 if b else 0

    print(f'File    : {args.input}')
    print(f'Counter : {method}')
    print()
    print(f"{'Format':<22} {'Chars':>8} {'Tokens':>8} {'vs JSON':>8} {'vs v1':>8}")
    print('-' * 58)
    print(f"{'JSON compact':<22} {len(json_compact):>8} {tc:>8}        -        -")
    print(f"{'JTF v1 (old)':<22} {len(jtf_v1_text):>8} {tv1:>8} {pct(tv1,tc):>+7.1f}%        -")
    print(f"{'JTF v2 (new)':<22} {len(jtf_v2_text):>8} {tv2:>8} {pct(tv2,tc):>+7.1f}% {pct(tv2,tv1):>+7.1f}%")


def main():
    p = argparse.ArgumentParser(
        prog='jtf',
        description='JTF v2: lossless JSON <-> compact token-efficient format'
    )
    sub = p.add_subparsers(dest='command', required=True)

    e = sub.add_parser('encode', help='JSON -> JTF')
    e.add_argument('input')
    e.add_argument('-o', '--output')

    d = sub.add_parser('decode', help='JTF -> JSON')
    d.add_argument('input')
    d.add_argument('-o', '--output')

    b = sub.add_parser('bench', help='Token comparison: JSON vs JTF v1 vs JTF v2')
    b.add_argument('input')

    args = p.parse_args()
    {'encode': cmd_encode, 'decode': cmd_decode, 'bench': cmd_bench}[args.command](args)


if __name__ == '__main__':
    main()
