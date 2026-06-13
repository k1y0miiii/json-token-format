#!/usr/bin/env python3
"""Shared conformance runner for JTF.

Runs BOTH the Python and JS reference libraries against every golden vector in
``conformance/cases/`` and reports per-language pass/fail counts. For each case
it checks both directions:

    encode(input)    == expected      (the .jtf file is the contract)
    decode(expected) == input         (lossless round-trip)

Exit code is non-zero if either language fails any case.

Usage:
    python3 conformance/run.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")
sys.path.insert(0, os.path.join(HERE, "..", "python", "src"))

from jtf import encode, decode  # noqa: E402


def _read_expected(file: str) -> str:
    with open(os.path.join(CASES, file), "r", encoding="utf-8") as f:
        raw = f.read()
    return raw[:-1] if raw.endswith("\n") else raw


def _load_manifest() -> dict:
    with open(os.path.join(HERE, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def run_python(manifest: dict):
    passed = 0
    failures = []
    for c in manifest["cases"]:
        with open(os.path.join(CASES, c["input"]), "r", encoding="utf-8") as f:
            data = json.load(f)
        expected = _read_expected(c["expected"])
        enc_ok = dec_ok = False
        note = ""
        try:
            got = encode(data)
            enc_ok = got == expected
            if not enc_ok:
                note = "encode mismatch"
            decoded = decode(expected)
            dec_ok = decoded == data
            if not dec_ok:
                note = (note + "; " if note else "") + "decode mismatch"
        except Exception as e:  # pragma: no cover
            note = f"error: {e}"
        if enc_ok and dec_ok:
            passed += 1
        else:
            failures.append((c["id"], c["name"], note))
    return passed, failures


def run_js(manifest: dict):
    """Invoke the JS runner; parse its per-line JSON results."""
    proc = subprocess.run(
        ["node", os.path.join(HERE, "runner_js.mjs")],
        capture_output=True,
        text=True,
    )
    passed = 0
    failures = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("encode_ok") and r.get("decode_ok"):
            passed += 1
        else:
            note = []
            if not r.get("encode_ok"):
                note.append("encode mismatch")
            if not r.get("decode_ok"):
                note.append("decode mismatch")
            if r.get("error"):
                note.append(r["error"].splitlines()[0])
            failures.append((r.get("id"), r.get("name"), "; ".join(note)))
    return passed, failures, proc.returncode, proc.stderr


def main() -> int:
    manifest = _load_manifest()
    total = len(manifest["cases"])
    print(f"JTF conformance suite — {total} vectors\n" + "=" * 48)

    py_pass, py_fail = run_python(manifest)
    print(f"\nPython : {py_pass}/{total} passed")
    for cid, name, note in py_fail:
        print(f"  FAIL  {cid}-{name}: {note}")

    js_pass, js_fail, js_rc, js_stderr = run_js(manifest)
    print(f"\nJS     : {js_pass}/{total} passed")
    for cid, name, note in js_fail:
        print(f"  FAIL  {cid}-{name}: {note}")

    print("\n" + "=" * 48)
    ok = (py_pass == total) and (js_pass == total)
    if ok:
        print(f"RESULT : PASS — both languages pass all {total} vectors.")
    else:
        print(
            f"RESULT : FAIL — Python {py_pass}/{total}, JS {js_pass}/{total}."
        )
        if js_stderr and js_pass == 0:
            print("JS stderr:\n" + js_stderr.strip())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
