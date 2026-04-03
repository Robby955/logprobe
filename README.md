# logprobe

Most logprob analysis pipelines silently assume normalized, complete distributions. In practice, top-k truncation and raw logits violate these assumptions, leading to systematically biased entropy and perplexity estimates.

logprobe detects these problems. It quantifies missing probability mass, catches unnormalized scores, and tells you exactly how much your entropy estimates are off.

## Real results on production API data

Tested on actual GPT-4o-mini and GPT-4.1-nano responses (April 2025). Raw JSON responses are in `demo/`.

**Creative writing** (GPT-4o-mini, temperature=0.7, top_logprobs=20):

```
$ logprobe diagnose demo/gpt4o_mini_creative.json

Normalization:  pass (log mass = -0.0119)
Missing mass:   0.0115 (150 positions)
Entropy bias:   +0.0155 bits (partial: 1.3450, normalized: 1.3605)
BPB:            byte data available

Validation: all validation checks passed (150 tokens)
```

Even with top-20, 1.15% of probability mass is missing. The entropy bias is small but measurable.

**GPT-2 with top-5** (the common case — most API calls default to top-5):

```
$ logprobe diagnose demo/gpt2_openai.json

Normalization:  pass (log mass = -0.5687)
Missing mass:   0.3001 (2/9 positions >50% missing)
Entropy bias:   +0.3743 bits (partial: 1.1134, normalized: 1.4876)
BPB:            byte data available

Validation: all validation checks passed (9 tokens)
```

30% of probability mass is missing. Two positions have >50% missing mass and are flagged UNRELIABLE.

**Raw logits passed as logprobs** (logprobe catches it immediately):

```
$ logprobe diagnose demo/gpt2_logits_openai.json

Normalization:  FAIL (log mass = 12.0840 — likely raw logits)
...
Validation: 63 error(s) found
  [ERROR] nonpositive_logprob (position 0): token " quick" has positive logprob 4.2831 ...
```

## The problem

LLM APIs return top-k logprobs for a position:

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

## Supported providers

logprobe auto-detects the input format. Use `--format <name>` to override.

| Provider | Format | Status |
|----------|--------|--------|
| **OpenAI** (GPT-4o, GPT-4.1, GPT-4.1-mini/nano) | `openai` | Full support (real API fixtures included) |
| **Azure OpenAI** | `openai` | Same format as OpenAI |
| **xAI / Grok** | `openai` | Same format as OpenAI |
| **Mistral AI** | `openai` | Same format as OpenAI |
| **DeepSeek** | `openai` | Same format as OpenAI (+ reasoning_content) |
| **Fireworks AI** | `openai` | Same format as OpenAI |
| **Google Gemini** | `gemini` | Native format (`logprobsResult`, `logProbability`) |
| **Ollama** (Llama, Gemma, Qwen, etc.) | `ollama` | Native format (top-level `logprobs` array) |
| **vLLM** (any model) | `vllm` | Flat token-array format |
| **Together AI** | `vllm` | Same format as vLLM |
| **HuggingFace TGI** | `openai` | OpenAI-compatible endpoint |
| **Amazon Bedrock** | `openai` | OpenAI-compatible mode (custom models only) |
| **JSONL / custom** | `jsonl` | One `{"token", "logprob"}` per line |

**Not supported** (no logprobs API): Anthropic/Claude, Groq, Perplexity, GPT-5/o1/o3 reasoning models.

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

The `demo/` directory contains real API responses and sample fixtures across multiple providers. See [demo/README.md](demo/README.md) for the full breakdown.

| Model | Task | Missing mass | Entropy bias | Perplexity |
|-------|------|-------------|-------------|------------|
| GPT-4o-mini | Creative (top-20, t=0.7) | 1.15% | +0.016 bits | 1.95 |
| GPT-4o-mini | Code (top-5, t=0) | ~0% | ~0 bits | 1.00 |
| GPT-4o-mini | Factual (top-5, t=0) | 0.10% | ~0 bits | 1.01 |
| GPT-2 | Scoring (top-5) | 30.0% | +0.374 bits | 5.39 |

## License

MIT
