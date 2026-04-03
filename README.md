# logprobe

Most logprob analysis pipelines silently assume normalized, complete distributions. In practice, top-k truncation and raw logits violate these assumptions, leading to systematically biased entropy and perplexity estimates.

logprobe detects these problems. It quantifies missing probability mass, catches unnormalized scores, and tells you exactly how much your entropy estimates are off.

## The problem

OpenAI returns top-5 logprobs for a position:

```
token       logprob     probability
"Hello"     -0.50       0.607
"Hi"        -2.00       0.135
"Hey"       -3.50       0.030
"Greetings" -4.10       0.017
"Good"      -4.80       0.008
                  total: 0.797
```

The observed tokens account for 79.7% of the probability mass. The remaining 20.3% is spread across thousands of unseen tokens. Renormalized top-k entropy is a lower bound on the true entropy (assuming correct top-k extraction) — but most tools present it as the actual value. If the API returned raw logits instead of log-probabilities (it happens), every metric silently produces garbage.

## What logprobe does

```
$ logprobe diagnose response.json

Normalization:  pass (log mass = -0.4285)
Missing mass:   0.3391 (2 positions)
Entropy bias:   -0.0966 bits (partial: 0.9504, normalized: 0.8539)
BPB:            byte data available

Validation: all validation checks passed (2 tokens)
```

One command. Tells you whether the data is usable, what's wrong, and how much error to expect.

**log mass** = `log(sum(exp(scores)))`. Approximately 0 for complete normalized distributions, negative for truncated top-k log-probabilities (expected), strongly positive for unnormalized logits (broken).

**Entropy bias** = `H_normalized - H_partial`. The sign is informative: positive means the missing tail inflates the entropy estimate, negative means top-k renormalization artificially concentrates mass. Large absolute values indicate significant truncation.

## Install

```
cargo install logprobe
```

## Commands

```
logprobe diagnose <file>      Detect normalization errors, entropy bias, and invalid distributions
logprobe validate <file>      Check logprob data integrity (finite, sorted, no duplicates, mass <= 1)
logprobe summary <file>       Sequence statistics (mean logprob, perplexity, missing mass)
logprobe entropy <file>       Per-token entropy from top_logprobs (partial and normalized)
logprobe confidence <file>    Find low-confidence tokens with surrounding context
logprobe bpb <file>           Bits-per-byte (strict: requires explicit byte counts)
logprobe highlight <file>     Terminal visualization colored by confidence
```

All commands read from stdin if no file is given. Add `--json` for machine-readable output.

## Formats

logprobe auto-detects three input formats:

- **OpenAI** — Chat Completions API response (`choices[0].logprobs.content`)
- **vLLM/Together** — Completions API response (`choices[0].logprobs.tokens` + `token_logprobs`)
- **JSONL** — One `{"token", "logprob"}` per line, or a JSON array of these objects

Use `--format openai|vllm|jsonl` to override detection, or `--strict-format` to reject ambiguous inputs.

## Why strict BPB

Most tools compute bits-per-byte as `-total_logprob / (total_bytes * ln(2))` where `total_bytes = sum(token.as_bytes().len())`. This is wrong for BPE tokenizers — tokens like `" Hello"` have a leading space byte that inflates the count, and special tokens have no meaningful byte representation at all.

logprobe refuses to compute BPB unless the API provides explicit byte arrays for each token. If your data doesn't include byte counts, logprobe tells you why instead of giving you a wrong number.

## Library usage

logprobe is also a library crate:

```rust
use logprobe::parse;
use logprobe::diagnostics;

let input = std::fs::read_to_string("response.json")?;
let seq = parse::parse_string(&input, None, false)?;

// Structured report
let report = diagnostics::diagnose_report(&seq);
println!("normalization: {:?}", report.normalization_status);
println!("mean missing mass: {:.4}", report.mean_missing_mass);
println!("entropy bias: {:+.4} bits", report.entropy_bias);

// Or flat findings list
let findings = diagnostics::diagnose(&seq);
for f in &findings {
    println!("[{:?}] {}: {}", f.severity, f.check, f.message);
}
```

## Demo

The `demo/` directory contains pre-generated GPT-2 logprob data in all three supported formats — including a deliberately corrupted file with raw logits instead of log-probabilities. No API key needed.

```bash
# Normal case — truncated top-5 from GPT-2
logprobe diagnose demo/gpt2_openai.json

# Broken case — raw logits. logprobe catches it immediately.
logprobe diagnose demo/gpt2_logits_openai.json
# → Normalization: FAIL (log mass = 12.0840 — likely raw logits)
# → Validation: 63 error(s) found
```

There's also a script to generate your own logprob data from GPT-2 locally. See [demo/README.md](demo/README.md) for details.

## License

MIT
