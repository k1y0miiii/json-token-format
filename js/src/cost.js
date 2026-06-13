// Token cost functions for JTF's value-dictionary break-even analysis.
//
// `heuristicCost` is the CANONICAL, deterministic cost function used by the
// conformance suite: max(1, floor(len/4)). It depends on nothing and yields
// byte-identical encoder output across languages and environments.
//
// For production tuning to a real GPT tokenizer, pass your own cost function
// (e.g. one backed by `gpt-tokenizer`) as the second arg to `encode`.

/** Deterministic token-count estimate: max(1, floor(len / 4)). */
export function heuristicCost(s) {
  return Math.max(1, Math.floor(s.length / 4));
}
