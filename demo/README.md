# logprobe demo

Pre-generated logprob data from GPT-2, plus a script to generate your own.
These files let you see what logprobe does without touching an API.

## Quick start (no setup needed)

The fixture files are already in this directory. Just run logprobe on them:

```bash
# 1. Normal case — truncated top-5 log-probabilities from a real model
logprobe diagnose demo/gpt2_openai.json
```

Expected output:
```
Normalization:  pass (log mass = -0.5687)
Missing mass:   0.3001 (2/9 positions >50% missing)
Entropy bias:   +0.3743 bits (partial: 1.1134, normalized: 1.4876)
BPB:            byte data available

Validation: all validation checks passed (9 tokens)
```

What this tells you:
- **Normalization pass** — scores are proper log-probabilities (log mass < 0, as expected for truncated top-k)
- **Missing mass 0.30** — on average, the top-5 tokens only cover 70% of the probability mass. The other 30% is spread across ~50,000 unseen tokens.
- **2 positions >50% missing** — at " quick" (position 0) and " lazy" (position 6), the model's distribution was so spread out that the top-5 captured less than half the probability. Entropy estimates at those positions are unreliable.
- **Entropy bias +0.37 bits** — renormalizing the truncated top-5 to sum to 1 inflates the entropy estimate by about 0.37 bits on average.

```bash
# 2. Broken case — raw logits (NOT log-probabilities). logprobe catches it.
logprobe diagnose demo/gpt2_logits_openai.json
```

Expected output:
```
Normalization:  FAIL (log mass = 12.0840 — likely raw logits)
Missing mass:   0.0000 (9 positions)
Entropy bias:   +11047975.3230 bits (partial: -11047973.7919, normalized: 1.5311)
BPB:            byte data available

Validation: 63 error(s) found
  [ERROR] nonpositive_logprob (position 0): token " quick" has positive logprob 4.2831 ...
  [ERROR] mass_exceeds_one (position 0): top_logprobs mass at position 0 is 4915.949726 ...
  ...
```

What this tells you:
- **Normalization FAIL** — log mass = 12.08 (should be approximately 0 or slightly negative). The scores are raw logits that were never softmaxed. Any perplexity, entropy, or BPB computed from these is **garbage**.
- **63 validation errors** — every token has a positive "logprob" (impossible for actual probabilities since log(p) <= 0), and every position has mass exceeding 1.0.
- This is exactly the kind of silent data corruption that logprobe is designed to catch. Most tools would happily compute perplexity from these numbers and give you a meaningless result.

```bash
# 3. vLLM format — same data, different API shape
logprobe diagnose demo/gpt2_vllm.json

# 4. JSONL stream — minimal format, no top_logprobs
logprobe diagnose demo/gpt2_stream.jsonl
```

The JSONL file has no top_logprobs, so logprobe can't check normalization or compute entropy bias — it tells you:
```
Normalization:  unknown (no top_logprobs data)
BPB:            byte data available

Validation: all validation checks passed (9 tokens)
```

## Other commands on the fixtures

```bash
# Per-token entropy breakdown
logprobe entropy demo/gpt2_openai.json

# Find low-confidence tokens with surrounding context
logprobe confidence demo/gpt2_openai.json --threshold -2.0

# Sequence summary (mean logprob, perplexity, missing mass)
logprobe summary demo/gpt2_openai.json

# BPB — works because byte arrays are included
logprobe bpb demo/gpt2_openai.json

# BPB — JSONL also has bytes
logprobe bpb demo/gpt2_stream.jsonl

# Terminal visualization (color-coded by confidence)
logprobe highlight demo/gpt2_openai.json

# JSON output for piping to other tools
logprobe diagnose demo/gpt2_openai.json --json
```

## Generate your own data

The `generate_logprobs.py` script runs GPT-2 locally and outputs logprob data in any format logprobe supports.

### Setup

```bash
pip install torch transformers
```

### Examples

```bash
# Teacher-forced scoring of fixed text (deterministic, reproducible)
python demo/generate_logprobs.py \
    --score "The quick brown fox jumps over the lazy dog." \
    --format openai \
    --output my_logprobs.json

# Then analyze with logprobe
logprobe diagnose my_logprobs.json

# Generate raw logits to see logprobe catch the problem
python demo/generate_logprobs.py \
    --score "The quick brown fox jumps over the lazy dog." \
    --format openai \
    --raw-logits \
    --output my_logits.json

logprobe diagnose my_logits.json
# → Normalization: FAIL (log mass = ... — likely raw logits)

# Autoregressive generation from a prompt
python demo/generate_logprobs.py \
    --prompt "Once upon a time" \
    --max-tokens 30 \
    --format openai

# vLLM format
python demo/generate_logprobs.py \
    --score "Hello world" \
    --format vllm

# JSONL stream
python demo/generate_logprobs.py \
    --score "Hello world" \
    --format jsonl

# Use GPU if available
python demo/generate_logprobs.py \
    --score "The answer to life is" \
    --device cuda
```

## What's in each file

| File | Format | What it demonstrates |
|------|--------|---------------------|
| `gpt2_openai.json` | OpenAI Chat Completions | Normal truncated top-5 logprobs with significant missing mass |
| `gpt2_logits_openai.json` | OpenAI (corrupted) | Raw logits passed as logprobs — logprobe catches this |
| `gpt2_vllm.json` | vLLM/Together Completions | Same data in flat token-array format |
| `gpt2_stream.jsonl` | JSONL | Minimal per-token logprobs, no top-k (BPB-only use case) |

All files score the text *"The quick brown fox jumps over the lazy dog."* using GPT-2 in teacher-forced mode. The values are realistic GPT-2 outputs where token confidence varies naturally — " over" and " the" are nearly certain while " quick" and " lazy" have substantial uncertainty.

## Verifying fixture consistency

```bash
python demo/verify_fixtures.py
```

Checks that all byte arrays match UTF-8 encodings, probabilities sum to <= 1 for normalized files, top_logprobs are sorted, and the logits file has the expected positive values.
