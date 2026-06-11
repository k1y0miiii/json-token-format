#!/usr/bin/env python3
"""
Benchmark: token counts (tiktoken cl100k_base) for compact JSON vs JTF v1 vs JTF v2.

Run with: cli-converter/venv/bin/python tests/benchmark.py
"""

import json
import os
import sys
import importlib.util
import pathlib

# Resolve paths
tests_dir = pathlib.Path(__file__).resolve().parent
cli_dir = tests_dir.parent

sys.path.insert(0, str(cli_dir))

import jtf as jtf_v2

# Load JTF v1
_spec = importlib.util.spec_from_file_location('jtf_v1', cli_dir / 'jtf_v1.py')
jtf_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jtf_v1)

# Load tiktoken
try:
    import tiktoken
    enc = tiktoken.get_encoding('cl100k_base')
    def count(s): return len(enc.encode(s))
    TOKENIZER = 'tiktoken cl100k_base'
except ImportError:
    def count(s): return max(1, len(s) // 4)
    TOKENIZER = 'heuristic (~4 chars/token) -- install tiktoken for accurate counts'

SAMPLE_FILES = [
    'config.json',
    'users_table.json',
    'api_response.json',
    'logs.json',
    'repeated_values.json',
    'nested_uniform.json',
]


def bench_file(path: pathlib.Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    compact = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    v1_text = jtf_v1.encode(data)
    v2_text = jtf_v2.encode(data)

    tc  = count(compact)
    tv1 = count(v1_text)
    tv2 = count(v2_text)

    return {
        'file': path.name,
        'compact': tc,
        'v1': tv1,
        'v2': tv2,
    }


def pct(a, b):
    if not b:
        return 0.0
    return (1 - a / b) * 100


def main():
    print(f'Tokenizer: {TOKENIZER}')
    print()

    col_w = [24, 8, 8, 8, 10, 10]
    headers = ['file', 'JSON', 'JTF v1', 'JTF v2', 'v2 vs JSON', 'v2 vs v1']
    row_fmt = (
        f'{{:<{col_w[0]}}} {{:>{col_w[1]}}} {{:>{col_w[2]}}} {{:>{col_w[3]}}}'
        f' {{:>{col_w[4]}}} {{:>{col_w[5]}}}'
    )

    sep = '-' * (sum(col_w) + len(col_w) - 1)
    print(row_fmt.format(*headers))
    print(sep)

    results = []
    for fname in SAMPLE_FILES:
        fpath = tests_dir / fname
        if not fpath.exists():
            print(f'  (skipping {fname}: not found)')
            continue
        r = bench_file(fpath)
        results.append(r)
        v2_vs_json = f'{pct(r["v2"], r["compact"]):+.1f}%'
        v2_vs_v1   = f'{pct(r["v2"], r["v1"]):+.1f}%'
        print(row_fmt.format(
            r['file'], r['compact'], r['v1'], r['v2'],
            v2_vs_json, v2_vs_v1
        ))

    if results:
        print(sep)
        # Totals
        tc  = sum(r['compact'] for r in results)
        tv1 = sum(r['v1'] for r in results)
        tv2 = sum(r['v2'] for r in results)
        print(row_fmt.format(
            'TOTAL', tc, tv1, tv2,
            f'{pct(tv2, tc):+.1f}%',
            f'{pct(tv2, tv1):+.1f}%'
        ))
        print()
        print('Positive % = tokens saved vs that baseline.')
        print('Negative % = regression vs that baseline.')

        regressions = [r for r in results if r['v2'] > r['compact']]
        if regressions:
            print()
            print('Files where JTF v2 is worse than compact JSON:')
            for r in regressions:
                diff = r['v2'] - r['compact']
                print(f"  {r['file']}: +{diff} tokens ({pct(r['v2'], r['compact']):+.1f}%)")
        else:
            print()
            print('JTF v2 is at least as good as compact JSON on all tested files.')


if __name__ == '__main__':
    main()
