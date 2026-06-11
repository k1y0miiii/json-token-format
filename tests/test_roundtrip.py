"""
Round-trip tests for JTF v2.

Covers:
  - All sample files (config, users_table, api_response, logs, repeated_values,
    nested_uniform)
  - Edge cases: null/true/false, empty string/object/array
  - Numeric-looking strings ("007", "3.14", "1e5", "true")
  - Strings containing = : tab newline comma quotes backslash cyrillic emoji
  - Deeply nested (4+ levels), arrays of arrays, mixed/heterogeneous arrays
  - Dictionary boundary cases ($1 vs $10 disambiguation)

Run with: cli-converter/venv/bin/python tests/test_roundtrip.py
"""

import json
import sys
import os

# Allow importing jtf from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jtf import encode, decode

PASS = 0
FAIL = 0


def check(name: str, data):
    global PASS, FAIL
    try:
        encoded = encode(data)
        decoded = decode(encoded)
        if decoded != data:
            print(f'FAIL  {name}')
            print(f'      original : {json.dumps(data, ensure_ascii=False)[:200]}')
            print(f'      decoded  : {json.dumps(decoded, ensure_ascii=False)[:200]}')
            print(f'      encoded  :\n{encoded[:400]}')
            FAIL += 1
        else:
            print(f'PASS  {name}')
            PASS += 1
    except Exception as e:
        print(f'ERROR {name}: {e}')
        import traceback; traceback.print_exc()
        FAIL += 1


def file_roundtrip(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    check(os.path.basename(path), data)


# ---- sample files ----
test_dir = os.path.dirname(os.path.abspath(__file__))
for fname in ('config.json', 'users_table.json', 'api_response.json',
              'logs.json', 'repeated_values.json', 'nested_uniform.json'):
    fpath = os.path.join(test_dir, fname)
    if os.path.exists(fpath):
        file_roundtrip(fpath)
    else:
        print(f'SKIP  {fname} (not found)')

# ---- primitives ----
check('null',         None)
check('true',         True)
check('false',        False)
check('int zero',     0)
check('negative int', -42)
check('float',        3.14)

# ---- numeric-looking strings (must round-trip as strings) ----
check('num-str 007',   '007')
check('num-str 3.14',  '3.14')
check('num-str 1e5',   '1e5')
check('num-str true',  {'flag': 'true'})
check('num-str false', {'flag': 'false'})
check('num-str null',  {'x': 'null'})
check('num-str -1',    {'-1': 'value', 'k': '-1'})
check('num-str 0.0',   '0.0')
check('num-str 1E10',  '1E10')

# ---- empty values ----
check('empty string',   '')
check('empty object',   {})
check('empty array',    [])
check('empty obj val',  {'a': {}})
check('empty arr val',  {'a': []})
check('empty nested',   {'a': {'b': {}}})

# ---- strings with special characters ----
check('string with =',        'key=value')
check('string with :',        'key: value')
check('string with tab',      'a\tb')
check('string with newline',  'line1\nline2')
check('string with comma',    'one, two, three')
check('string with quote',    'say "hello"')
check('string with backslash','C:\\Users\\test')
check('string starting with #', '#hashtag')
check('string starting with $', '$money')
check('string with equals at end', {'k': 'val='})
check('cyrillic',             'Привет мир')
check('emoji',                'test☃')  # snowman
check('mixed cyrillic emoji', {'msg': 'Привет \U0001F600'})

# ---- objects ----
check('flat object', {
    'name': 'Иван', 'age': 30, 'active': True, 'score': 9.5, 'notes': None
})
check('nested object', {
    'app': 'test',
    'db': {
        'host': 'localhost',
        'port': 5432,
        'pool': {'min': 2, 'max': 10}
    }
})
check('deeply nested 4 levels', {
    'l1': {'l2': {'l3': {'l4': {'value': 'deep', 'n': 42}}}}
})
check('deeply nested 5 levels', {
    'a': {'b': {'c': {'d': {'e': 'bottom'}}}}
})
check('object with reserved-word keys', {
    'null': 1, 'true': 2, 'false': 3
})
check('object with = in key', {'ke=y': 'val'})
check('object with : in key', {'ke:y': 'val'})
check('object with tab in key', {'ke\ty': 'val'})

# ---- arrays ----
check('primitive int array', [1, 2, 3, 4, 5])
check('primitive string array', ['alpha', 'beta', 'gamma'])
check('mixed primitive array', [1, 'two', None, True, 3.14])
check('array of arrays', [[1, 2], [3, 4], [5, 6]])
check('array of arrays nested', [[1, [2, 3]], [4, [5, 6]]])
check('mixed/heterogeneous array', [
    {'a': 1},
    {'a': 1, 'b': 2},
    'plain string',
    42,
    None,
])
check('uniform flat array', [
    {'id': 1, 'name': 'Alice', 'active': True},
    {'id': 2, 'name': 'Bob',   'active': False},
    {'id': 3, 'name': 'Клод', 'active': True},
])
check('uniform array with nulls', [
    {'x': 1, 'y': None, 'z': 'a'},
    {'x': 2, 'y': 99,   'z': 'b'},
])
check('uniform array cyrillic keys', [
    {'id': 1, 'имя': 'Алексей', 'роль': 'admin'},
    {'id': 2, 'имя': 'Мария',   'роль': 'editor'},
])
check('nested uniform array', [
    {'id': 1, 'addr': {'city': 'Moscow',   'zip': '101000'}},
    {'id': 2, 'addr': {'city': 'Kazan',    'zip': '420000'}},
    {'id': 3, 'addr': {'city': 'Samara',   'zip': '443000'}},
])
check('deeply nested uniform array', [
    {'id': i, 'meta': {'stats': {'score': float(i), 'rank': i}}}
    for i in range(5)
])
check('array of empty objects', [{}, {}, {}])
check('single-element array', [42])
check('single-element obj array', [{'x': 1}])
check('large uniform table', [
    {'id': i, 'val': f'item_{i}', 'score': float(i * 1.5), 'ok': i % 2 == 0}
    for i in range(20)
])

# ---- nested containers in objects ----
check('nested array in object', {
    'users': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}],
    'count': 2
})
check('array of arrays in object', {'matrix': [[1, 0], [0, 1]]})
check('url in value', {
    'url': 'https://example.com/path?q=1&r=2',
    'title': 'Example'
})
check('timestamps', {
    'created': '2024-11-18T14:32:07Z',
    'updated': '2024-11-19T00:00:00Z'
})

# ---- value dictionary boundary cases ----
# These generate many dict entries; test that $1 vs $10 round-trips correctly.
check('dict boundary $1 vs $10', {
    'v0': 'aaaa_aaaa_aaaa_aaaa',
    'v1': 'bbbb_bbbb_bbbb_bbbb',
    'v2': 'cccc_cccc_cccc_cccc',
    'v3': 'dddd_dddd_dddd_dddd',
    'v4': 'eeee_eeee_eeee_eeee',
    'v5': 'ffff_ffff_ffff_ffff',
    'v6': 'gggg_gggg_gggg_gggg',
    'v7': 'hhhh_hhhh_hhhh_hhhh',
    'v8': 'iiii_iiii_iiii_iiii',
    'v9': 'jjjj_jjjj_jjjj_jjjj',
    'v10': 'kkkk_kkkk_kkkk_kkkk',  # $10 should not be mis-parsed as $1 + "0"
    'r0': 'aaaa_aaaa_aaaa_aaaa',
    'r1': 'bbbb_bbbb_bbbb_bbbb',
    'r2': 'cccc_cccc_cccc_cccc',
    'r3': 'dddd_dddd_dddd_dddd',
    'r4': 'eeee_eeee_eeee_eeee',
    'r5': 'ffff_ffff_ffff_ffff',
    'r6': 'gggg_gggg_gggg_gggg',
    'r7': 'hhhh_hhhh_hhhh_hhhh',
    'r8': 'iiii_iiii_iiii_iiii',
    'r9': 'jjjj_jjjj_jjjj_jjjj',
    'r10': 'kkkk_kkkk_kkkk_kkkk',
})
check('dict value with = sign', [
    {'status': 'key=value', 'other': 'key=value'},
    {'status': 'key=value', 'other': 'different'},
])
check('dict value that looks numeric', [
    {'v': '3.14', 'other': '3.14'},
    {'v': '3.14', 'other': 'something'},
])
check('no dict when not profitable', {'x': 'ab', 'y': 'ab'})

# ---- misc ----
check('top-level primitive string', 'hello world')
check('top-level primitive int', 42)
check('top-level primitive null', None)
check('top-level primitive bool', True)
check('top-level primitive float', 3.14)

print()
print(f'Results: {PASS} passed, {FAIL} failed')
sys.exit(0 if FAIL == 0 else 1)
