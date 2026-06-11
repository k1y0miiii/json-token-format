#!/usr/bin/env python3
"""
JTF - JSON Token Format
A compact, lossless, round-trippable encoding of JSON data for LLM prompts.

Format spec (v1):

  Primitives:
    null, true, false written literally.
    Numbers written in canonical JSON form (no trailing zeros).
    Strings: unquoted when safe (no comma, colon, pipe, newline, leading/trailing space,
    no collision with null/true/false, no starting with " [ { - #).
    Unsafe strings are double-quoted with JSON escape sequences (handles Cyrillic fine).

  Objects (at indent level N, using 2 spaces per level):
    key: value         <- primitive or inline array
    key:               <- followed by indented child block
      child_key: ...

  Primitive arrays:
    [N]: v1, v2, v3    <- inline

  Uniform object arrays (all dicts, same keys, all primitive values):
    [N]{k1,k2,k3}:
      v1,v2,v3
      v4,v5,v6

  Mixed/irregular arrays:
    [N]:
      - item1
      - item2
      - nested_key: val
        another_key: val

  Top-level value is written at indent 0 with no enclosing braces.
"""

import json
import sys
import argparse
import re

# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r'^[^\n\r,:|"\\#\[\]{}\t]+$')
_RESERVED = frozenset({'null', 'true', 'false'})


def _safe(s: str) -> bool:
    if not s:
        return False
    if s in _RESERVED:
        return False
    if s != s.strip():
        return False
    if s[0] in ('"', '[', '{', '-', '#'):
        return False
    if ': ' in s or s.endswith(':'):
        return False
    return bool(_SAFE_RE.match(s))


def _es(s: str) -> str:
    """Encode a string token."""
    return s if _safe(s) else json.dumps(s, ensure_ascii=False)


def _ep(v) -> str:
    """Encode a primitive value."""
    if v is None:
        return 'null'
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return _es(v)


def _is_prim(v) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _uniform(arr: list) -> bool:
    """True if arr is a non-empty list of dicts with same keys and all-primitive values."""
    if not arr or not isinstance(arr[0], dict) or not arr[0]:
        return False
    keys = list(arr[0].keys())
    ks = set(keys)
    for item in arr:
        if not isinstance(item, dict) or set(item.keys()) != ks:
            return False
        if not all(_is_prim(v) for v in item.values()):
            return False
    return True


def _encode_lines(val, depth: int) -> list[str]:
    """
    Returns a list of strings (lines) representing val at the given depth.
    Lines do NOT include a leading newline; callers concatenate with '\\n'.
    The returned lines are already indented at 'depth' spaces where appropriate.
    """
    pad = '  ' * depth

    if _is_prim(val):
        return [_ep(val)]

    if isinstance(val, list):
        return _encode_array_lines(val, depth)

    if isinstance(val, dict):
        return _encode_obj_lines(val, depth)

    raise TypeError(f"Unsupported type: {type(val)}")


def _encode_array_lines(arr: list, depth: int) -> list[str]:
    pad = '  ' * depth
    n = len(arr)

    if n == 0:
        return ['[0]:']

    if all(_is_prim(x) for x in arr):
        inner = ', '.join(_ep(x) for x in arr)
        return [f'[{n}]: {inner}']

    if _uniform(arr):
        keys = list(arr[0].keys())
        hdr_keys = ','.join(_es(k) for k in keys)
        lines = [f'[{n}]{{{hdr_keys}}}: ']
        # Trim the trailing space from header
        lines[0] = lines[0].rstrip()
        row_pad = pad + '  '
        for item in arr:
            cells = ','.join(_ep(item[k]) for k in keys)
            lines.append(row_pad + cells)
        return lines

    # Mixed / dash list
    lines = [f'[{n}]:']
    for item in arr:
        item_lines = _encode_item_lines(item)
        if item_lines:
            lines.append(pad + '  - ' + item_lines[0])
            for subsequent in item_lines[1:]:
                lines.append(pad + '    ' + subsequent)
        else:
            lines.append(pad + '  -')
    return lines


def _encode_item_lines(val) -> list[str]:
    """Return lines for a dash-list item at depth 0 (no leading indent)."""
    if _is_prim(val):
        return [_ep(val)]
    if isinstance(val, list):
        return _encode_array_lines(val, 0)
    if isinstance(val, dict):
        return _encode_obj_lines(val, 0)
    raise TypeError(f"Unsupported type: {type(val)}")


def _encode_obj_lines(obj: dict, depth: int) -> list[str]:
    if not obj:
        return ['{}']
    pad = '  ' * depth
    lines = []
    for k, v in obj.items():
        ek = _es(k)
        if _is_prim(v):
            lines.append(pad + f'{ek}: {_ep(v)}')
        elif isinstance(v, list):
            arr_lines = _encode_array_lines(v, depth)
            if len(arr_lines) == 1:
                lines.append(pad + f'{ek}: {arr_lines[0]}')
            else:
                # First line goes after the key
                lines.append(pad + f'{ek}: {arr_lines[0]}')
                lines.extend(arr_lines[1:])
        elif isinstance(v, dict):
            if not v:
                lines.append(pad + f'{ek}: {{}}')
            else:
                lines.append(pad + f'{ek}:')
                lines.extend(_encode_obj_lines(v, depth + 1))
        else:
            raise TypeError(f"Unsupported type: {type(v)}")
    return lines


def encode(data) -> str:
    """Encode a Python JSON-compatible object to JTF string."""
    if isinstance(data, dict):
        lines = _encode_obj_lines(data, 0)
    elif isinstance(data, list):
        lines = _encode_array_lines(data, 0)
    elif _is_prim(data):
        lines = [_ep(data)]
    else:
        raise TypeError(f"Unsupported top-level type: {type(data)}")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class _Lines:
    def __init__(self, text: str):
        raw = text.splitlines()
        # Keep lines; track their indent
        self.lines = raw
        self.pos = 0

    def eof(self) -> bool:
        return self.pos >= len(self.lines)

    def peek_raw(self) -> str | None:
        if self.pos >= len(self.lines):
            return None
        return self.lines[self.pos]

    def peek(self) -> tuple[int, str] | None:
        """Return (indent, stripped_content) of next non-blank line without consuming."""
        i = self.pos
        while i < len(self.lines):
            raw = self.lines[i]
            stripped = raw.rstrip()
            if stripped.strip():
                ind = len(stripped) - len(stripped.lstrip(' '))
                return (ind, stripped.strip())
            i += 1
        return None

    def consume(self) -> tuple[int, str]:
        """Consume and return (indent, stripped_content)."""
        while self.pos < len(self.lines):
            raw = self.lines[self.pos]
            self.pos += 1
            stripped = raw.rstrip()
            if stripped.strip():
                ind = len(stripped) - len(stripped.lstrip(' '))
                return (ind, stripped.strip())
        raise EOFError("No more lines")


# CSV split respecting double-quoted fields
def _csv_split(s: str) -> list[str]:
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


def _dec_str(tok: str) -> str:
    tok = tok.strip()
    if tok.startswith('"'):
        return json.loads(tok)
    return tok


def _dec_prim(tok: str):
    tok = tok.strip()
    if tok == 'null':  return None
    if tok == 'true':  return True
    if tok == 'false': return False
    if tok == '{}':    return {}
    if tok == '[]':    return []
    if tok.startswith('"'):
        return json.loads(tok)
    # Number?
    try:
        if '.' in tok or 'e' in tok.lower():
            return float(tok)
        return int(tok)
    except ValueError:
        pass
    # Bare string
    return tok


_ARR_HEADER = re.compile(r'^\[(\d+)\](\{[^}]*\})?:(.*)$')
_KEY_VALUE   = re.compile(r'^([^:\[\]]+?)\s*:\s*(.*)$')


def _parse_value(lines: _Lines, min_indent: int):
    """Parse one value. min_indent is the indent level we expect content at."""
    p = lines.peek()
    if p is None:
        return None
    ind, content = p

    # Empty object/array literals
    if content in ('{}', '[]', '[0]:'):
        lines.consume()
        if content == '{}': return {}
        return []

    # Array header?
    m = _ARR_HEADER.match(content)
    if m:
        return _parse_array(lines, ind)

    # Try object (multiple key:value lines at same indent)
    if _looks_like_obj_line(content):
        return _parse_object(lines, ind)

    # Scalar
    ind2, content2 = lines.consume()
    return _dec_prim(content2)


def _looks_like_obj_line(content: str) -> bool:
    """Does this look like a key: value or key: line?"""
    if content.startswith('[') or content.startswith('-') or content.startswith('"'):
        return False
    if _ARR_HEADER.match(content):
        return False
    # Check for key: pattern
    m = re.match(r'^([^:\[\]]+?)\s*:', content)
    if not m:
        return False
    # Exclude bare primitives that happen to contain colons (e.g. timestamps)
    key_candidate = m.group(1).strip()
    # If key_candidate looks like it contains spaces and is a long string, probably not a key
    # Simple heuristic: keys shouldn't contain certain chars
    if any(c in key_candidate for c in [',', '[', ']', '{', '}']):
        return False
    # A URL like https://... would have // after colon which is fine as a value
    return True


def _parse_array(lines: _Lines, arr_indent: int) -> list:
    ind, content = lines.consume()
    m = _ARR_HEADER.match(content)
    if not m:
        raise ValueError(f"Expected array header, got: {content!r}")

    count = int(m.group(1))
    fields_str = m.group(2)  # e.g. '{k1,k2,k3}' or None
    rest = m.group(3).strip()

    if count == 0:
        return []

    # Tabular: [N]{fields}:
    if fields_str:
        fields = [_dec_str(f) for f in _csv_split(fields_str[1:-1])]
        rows = []
        for _ in range(count):
            _, row_content = lines.consume()
            cells = _csv_split(row_content)
            if len(cells) != len(fields):
                raise ValueError(
                    f"Expected {len(fields)} cells, got {len(cells)}: {row_content!r}"
                )
            rows.append({f: _dec_prim(c) for f, c in zip(fields, cells)})
        return rows

    # Inline primitive: [N]: v1, v2, ...
    if rest:
        return [_dec_prim(x) for x in _csv_split(rest)]

    # Dash list: [N]:\n  - item\n  - item
    items = []
    for _ in range(count):
        p = lines.peek()
        if p is None:
            raise ValueError("Unexpected EOF in dash list")
        item_ind, item_content = p
        if not item_content.startswith('-'):
            raise ValueError(f"Expected '- item', got: {item_content!r}")
        items.append(_parse_dash_item(lines, item_ind))
    return items


def _parse_dash_item(lines: _Lines, item_indent: int):
    ind, content = lines.consume()
    if content == '-':
        return None
    if content.startswith('- '):
        rest = content[2:].strip()
    elif content.startswith('-'):
        rest = content[1:].strip()
    else:
        raise ValueError(f"Expected dash item: {content!r}")

    if not rest:
        return None

    # Check if rest is an array header
    m = _ARR_HEADER.match(rest)
    if m:
        # Insert back as a line and parse
        lines.lines.insert(lines.pos, ' ' * (item_indent + 2) + rest)
        return _parse_array(lines, item_indent + 2)

    # Check if rest is an object (key: value)
    if _looks_like_obj_line(rest):
        # Insert back as a line and parse object starting at item_indent+2
        lines.lines.insert(lines.pos, ' ' * (item_indent + 2) + rest)
        return _parse_object(lines, item_indent + 2)

    return _dec_prim(rest)


def _parse_object(lines: _Lines, obj_indent: int) -> dict:
    obj = {}
    while True:
        p = lines.peek()
        if p is None:
            break
        ind, content = p
        if ind < obj_indent:
            break
        if ind > obj_indent:
            break

        # Must be a key: value or key: line
        if not _looks_like_obj_line(content):
            break

        ind2, content2 = lines.consume()

        # Find the colon separating key from value
        colon = _find_colon(content2)
        if colon == -1:
            # Shouldn't happen given _looks_like_obj_line, treat as string
            break

        raw_key = content2[:colon].strip()
        key = _dec_str(raw_key)
        rest = content2[colon+1:].strip()

        if not rest:
            # Value on next lines
            p2 = lines.peek()
            if p2 is None:
                obj[key] = None
                continue
            child_ind, child_content = p2
            if child_ind <= obj_indent:
                obj[key] = None
                continue
            obj[key] = _parse_value(lines, child_ind)
        else:
            # Inline or array header
            m = _ARR_HEADER.match(rest)
            if m:
                # Insert back inline and parse
                lines.lines.insert(lines.pos, ' ' * (obj_indent + 2) + rest)
                obj[key] = _parse_array(lines, obj_indent + 2)
            elif _looks_like_obj_line(rest) and not _is_likely_prim(rest):
                # Nested single-line object? Unusual, parse as string
                obj[key] = _dec_prim(rest)
            else:
                obj[key] = _dec_prim(rest)

    return obj


def _is_likely_prim(s: str) -> bool:
    """Heuristic: is this string more likely a primitive than a key:value?"""
    if s in ('null', 'true', 'false'):
        return True
    if s.startswith('"'):
        return True
    try:
        float(s.split()[0])
        return True
    except (ValueError, IndexError):
        pass
    return False


def _find_colon(s: str) -> int:
    """Find first unquoted ':' that is a key separator."""
    in_q = False
    for i, c in enumerate(s):
        if c == '"' and (i == 0 or s[i-1] != '\\'):
            in_q = not in_q
        if not in_q and c == ':':
            return i
    return -1


def decode(text: str):
    """Decode a JTF string to Python object."""
    lines = _Lines(text)
    return _parse_value(lines, 0)


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
        print(f"Encoded → {args.output}")
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
        print(f"Decoded → {args.output}")
    else:
        print(result)


def cmd_bench(args):
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    json_compact = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    json_pretty  = json.dumps(data, ensure_ascii=False, indent=2)
    jtf_text     = encode(data)

    try:
        import tiktoken
        enc_tok = tiktoken.get_encoding("cl100k_base")
        def count(s): return len(enc_tok.encode(s))
        method = "tiktoken cl100k_base"
    except Exception:
        def count(s): return max(1, len(s) // 4)
        method = "heuristic (~4 chars/token)"

    tc  = count(json_compact)
    tp  = count(json_pretty)
    tj  = count(jtf_text)

    sv_compact = (1 - tj / tc) * 100 if tc else 0
    sv_pretty  = (1 - tj / tp) * 100 if tp else 0

    print(f"File    : {args.input}")
    print(f"Counter : {method}")
    print()
    print(f"{'Format':<22} {'Chars':>8} {'Tokens':>8}")
    print('-' * 42)
    print(f"{'JSON compact':<22} {len(json_compact):>8} {tc:>8}")
    print(f"{'JSON pretty':<22} {len(json_pretty):>8} {tp:>8}")
    print(f"{'JTF':<22} {len(jtf_text):>8} {tj:>8}")
    print()
    print(f"JTF vs JSON compact: {sv_compact:+.1f}%")
    print(f"JTF vs JSON pretty : {sv_pretty:+.1f}%")


def main():
    p = argparse.ArgumentParser(
        prog='jtf',
        description='JTF: lossless JSON <-> compact token-efficient format'
    )
    sub = p.add_subparsers(dest='command', required=True)

    e = sub.add_parser('encode', help='JSON → JTF')
    e.add_argument('input')
    e.add_argument('-o', '--output')

    d = sub.add_parser('decode', help='JTF → JSON')
    d.add_argument('input')
    d.add_argument('-o', '--output')

    b = sub.add_parser('bench', help='Token comparison: JSON vs JTF')
    b.add_argument('input')

    args = p.parse_args()
    {'encode': cmd_encode, 'decode': cmd_decode, 'bench': cmd_bench}[args.command](args)


if __name__ == '__main__':
    main()
