#!/usr/bin/env node
// JTF CLI (JS). Mirrors the Python CLI.
//
//   jtf encode input.json [-o out.jtf]   (use '-' to read stdin)
//   jtf decode input.jtf  [-o out.json]

import { readFileSync, writeFileSync } from "node:fs";
import { encode, decode } from "../src/jtf.js";

function readInput(path) {
  if (path === "-") return readFileSync(0, "utf8");
  return readFileSync(path, "utf8");
}

function writeOutput(path, text) {
  if (path) writeFileSync(path, text + "\n");
  else process.stdout.write(text + "\n");
}

function parseArgs(argv) {
  const [cmd, ...rest] = argv;
  let input = null;
  let output = null;
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    if (a === "-o" || a === "--output") output = rest[++i];
    else input = a;
  }
  return { cmd, input, output };
}

function main() {
  const { cmd, input, output } = parseArgs(process.argv.slice(2));
  if (!cmd || (cmd !== "encode" && cmd !== "decode") || !input) {
    process.stderr.write(
      "usage: jtf encode|decode <input> [-o output]\n" +
        "       use '-' as <input> to read stdin\n"
    );
    process.exit(2);
  }
  if (cmd === "encode") {
    const data = JSON.parse(readInput(input));
    writeOutput(output, encode(data));
  } else {
    const data = decode(readInput(input));
    writeOutput(output, JSON.stringify(data, null, 2));
  }
}

main();
