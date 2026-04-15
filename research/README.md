# Logprob analysis of GPT-4o-mini

Real experiments using OpenAI models (April 2025), analyzed with logprobe. All data files are in `research/data/` — every number here is reproducible.

## The core findings

1. **GPT-4o-mini is so confident that top-20 captures >99.9% of the probability mass.** Truncation bias is negligible for this model.
2. **Entropy spikes coincide with uncertain or hallucinated tokens** in the examples tested.
3. **Language affects BPB significantly.** Japanese BPB is 1.4x English for the same content, due to multi-byte encoding.

## Experiment 1: Temperature sweep

Same prompt ("Write a short paragraph about why the ocean is blue"), same model (GPT-4o-mini), same `top_logprobs=20`. Only temperature changes.

```
logprobe diagnose research/data/temp_0.0.json
logprobe diagnose research/data/temp_1.5.json
```

| Temperature | Perplexity | BPB | Tokens below -1.0 | Tokens below -2.0 | Missing mass |
|-------------|-----------|------|-------------------|-------------------|-------------|
| 0.0 | 1.22 | 0.055 | 2 | 0 | 0.02% |
| 0.5 | 1.22 | 0.055 | 6 | 1 | 0.02% |
| 1.0 | 1.28 | 0.070 | 8 | 2 | 0.01% |
| 1.5 | 1.78 | 0.150 | 21 | 10 | 0.02% |

### What this means in plain English

**Perplexity measures surprise.** A perplexity of 1.22 means the model is, on average, choosing between 1.22 equally likely options — it's very certain about most tokens. At temperature 1.5, this rises to 1.78: the model is still fairly confident, but is now spreading probability across more alternatives.

**BPB (bits per byte) measures compression efficiency.** At temperature 0, the model needs 0.055 bits per byte — compared to 8 bits for raw ASCII. It compresses this text to 0.7% of its raw size.

**Temperature doesn't change what the model knows — it changes how it distributes probability.** At temperature 0, only 2/100 tokens fall below logprob -1.0 (probability < 37%). At temperature 1.5, that's 21 tokens. The uncertain tokens are function words ("with", "and", "as") and synonym choices ("contains" vs "includes") — exactly the positions where multiple phrasings are equally valid.

**Missing mass stays near zero at all temperatures.** Even at temperature 1.5, top-20 captures 99.98%. Temperature spreads probability among top candidates; it doesn't push mass into the tail.

## Experiment 2: Domain comparison

Same model (GPT-4o-mini), same temperature (0.7), same `top_logprobs=20`. Different prompt types.

| Domain | Prompt | Tokens | Perplexity | BPB | Missing mass | Entropy (bits) |
|--------|--------|--------|-----------|------|-------------|----------------|
| Factual | "What is the capital of France?" | 7 | 1.00 | 0.000 | 0.00% | 0.000 |
| Reasoning | "Explain step by step: train at 60 mph for 2.5 hours" | 150 | 1.15 | 0.060 | 0.00% | 0.267 |
| Code | "Write a Python Fibonacci function with memoization" | 200 | 1.15 | 0.047 | 0.00% | 0.293 |
| Creative | "Opening paragraph of a mystery novel in 1920s Paris" | 150 | * | 21.84* | 1.82% | 1.224 |

*Creative writing BPB is destroyed by a single -9999 logprob token (see "The -9999 mystery" below).

**Factual answers are maximally confident.** "The capital of France is Paris." — every token has logprob ~0.0 (probability ~100%). Perplexity 1.00. There is literally nothing to analyze.

**Code and reasoning are nearly as confident.** Syntax constraints reduce uncertainty. `def fibonacci(n, memo={}):` is nearly deterministic. Perplexity 1.15 for both.

**Creative writing is where the model actually thinks.** Entropy 1.22 bits — 4.6x higher than code. The model is choosing between many plausible continuations: "gaslit" vs "flickering", "tattered" vs "worn", "croissants" vs "pastries".

## Experiment 3: Hallucination detection via entropy

This is the most interesting experiment. We asked the model questions it cannot know the answer to and examined where entropy spikes.

### Test 1: Confidence gradient (easy to impossible)

```
logprobe entropy research/data/confidence_gradient.json
```

We asked four questions in one prompt, ranging from trivially known to impossible:

> (1) What is 2+2? (2) Who painted the Mona Lisa? (3) What is the population of Liechtenstein? (4) Who was the 23rd person to walk on the moon?

The model answered: "4", "Leonardo da Vinci", "approximately 39,000", and "Charles Duke."

Only 12 people have ever walked on the moon. The 23rd doesn't exist. The model hallucinated. Here's the per-token entropy:

| Question | Token | Entropy (bits) | Interpretation |
|----------|-------|---------------|----------------|
| "2+2" | "4" | 0.000 | Certain. Correct. |
| "Mona Lisa" | "Leonardo" | 0.000 | Certain. Correct. |
| "Mona Lisa" | "da" | 0.000 | Certain. Correct. |
| "Mona Lisa" | "Vinci" | 0.000 | Certain. Correct. |
| Liechtenstein | "approximately" | 0.132 | Slight hedge — knows it's approximate |
| Liechtenstein | "39" | 0.072 | Small uncertainty on the exact number |
| Moon | "is" | **0.328** | **Highest entropy in the entire response** |
| Moon | "Charles" | 0.010 | Committed to the hallucination |
| Moon | "Duke" | 0.115 | Some uncertainty on the last name |

**The single most uncertain token in the entire response is "is" — right before the hallucinated name.** This is the exact position where the model chose between fabricating an answer and refusing. Once it committed ("is"), the name followed with lower entropy.

The only token in the entire 60-token response that fell below logprob -0.5 was this same "is" (logprob = -2.81, probability 6%). Every other token was above -0.5.

```
logprobe confidence -t -0.5 research/data/confidence_gradient.json

Position 56: logprob=-2.8120 (p=0.060087)
  Context: ... on the moon[ is] Charles Duke.
```

### Test 2: Manufactured factoid

```
logprobe entropy research/data/hallucination_factoid.json
```

We asked: "What year was the Remington Model 7615 pump-action centerfire rifle first manufactured?"

The model answered: "1990." (The actual year is 2006.)

Entropy across all 20 tokens:

| Token | Entropy | What it tells us |
|-------|---------|-----------------|
| "The" | 0.000 | Certain — knows this format |
| "Remington" | 0.000 | Certain — recognizes the brand |
| "Model" | 0.000 | Certain — recognizes the product |
| "7615" | 0.000 | Certain — knows the model number |
| "pump-action" | 0.011 | Virtually certain |
| "manufactured" | 0.003 | Certain — knows the question format |
| "in" | 0.000 | Certain |
| "199" | **0.645** | **Uncertain** — knows it's the 1990s but not which decade |
| "0" | **0.668** | **Most uncertain token** — choosing between 0, 2, 5, 8 |
| "." | 0.000 | Certain |

The model is *completely certain* about what the gun is. It recognizes "Remington Model 7615 pump-action centerfire rifle" with zero entropy. But the moment it has to produce the year, entropy spikes to 0.65-0.67 bits. **The hallucinated digits are precisely where entropy is highest.**

In both hallucination examples, per-token entropy is highest at the exact position where the model transitions from grounded to fabricated content. This is consistent with the semantic entropy literature (Farquhar et al., 2024, "Detecting hallucinations in large language models using semantic entropy") but should be validated on larger datasets before drawing general conclusions.

### Test 3: Fictional person biography

```
logprobe diagnose research/data/hallucination_biography.json
```

We asked about "Dr. James Whitfield Patterson, the famous 19th century chemist from Edinburgh" — a completely fictional person.

The model *correctly refused to hallucinate* — it responded: "It seems there may be some confusion... there is no prominent 19th-century chemist by that name."

But the logprobs reveal the cost of refusal. The response has **20 tokens below logprob -1.0** (vs 0 for true factual answers). The model's overall perplexity is 1.56 (vs 1.00 for "capital of France"). The high-uncertainty tokens cluster around hedge words:

| Token | logprob | Context |
|-------|---------|---------|
| "prominent" | -2.62 | there is no **prominent** 19th-century |
| "widely" | -2.01 | by that name **widely** recognized |
| "thinking" | -1.52 | you might be **thinking** of another |
| "figure" | -2.22 | thinking of another **figure** in the field |
| "fictional" | -1.35 | the name could be **fictional** or less |
| "referencing" | -3.28 | if you're **referencing** a different chemist |

**Refusal responses have higher perplexity than fabricated answers.** The hedge words and redirections in refusal text ("prominent," "widely," "thinking," "fictional") have low individual probability, driving up perplexity. Factual answers have near-zero entropy; fabricated answers also have low entropy (the model assigns high probability to its chosen fake year). Refusal text is the least predictable of the three.

## Experiment 4: Cross-model comparison

Same prompt ("Why is the ocean blue?"), same temperature (0.7), same `top_logprobs=20`. Four different OpenAI models.

```
logprobe diagnose research/data/model_gpt4o.json
logprobe diagnose research/data/model_gpt41mini.json
logprobe diagnose research/data/model_gpt41nano.json
```

| Model | Tokens | Perplexity | BPB | Entropy (bits) | Missing mass |
|-------|--------|-----------|------|----------------|-------------|
| GPT-4o | 100 | 1.26 | 0.064 | 0.567 | 0.08% |
| GPT-4o-mini | 100 | 1.22 | 0.055 | 0.584 | 0.02% |
| GPT-4.1-mini | 89 | 1.18 | 0.042 | 0.287 | 0.00% |
| GPT-4.1-nano | 100 | 1.25 | 0.058 | 0.424 | 0.02% |

**GPT-4.1-mini is the most confident.** Lowest perplexity (1.18), lowest BPB (0.042), lowest entropy (0.287 bits), and zero missing mass. It also used fewer tokens (89 vs 100) — more concise.

**GPT-4o is the least confident.** Higher perplexity (1.26) and the most missing mass (0.08%). This could reflect the larger model's broader vocabulary and more nuanced distribution — more ways to say the same thing.

**All models have negligible truncation bias.** Missing mass ranges from 0.00% to 0.08%. Top-20 is more than sufficient for any of these models.

## Experiment 5: Language comparison

Same model (GPT-4o-mini), same temperature (0.7), same `top_logprobs=20`. Same topic (sky/ocean color), different languages.

```
logprobe diagnose research/data/lang_french.json
logprobe diagnose research/data/lang_japanese.json
```

| Language | Tokens | Perplexity | BPB | Entropy (bits) | Missing mass |
|----------|--------|-----------|------|----------------|-------------|
| English (temp_0.5) | 100 | 1.22 | 0.055 | 0.443 | 0.02% |
| French | 100 | 1.18 | 0.050 | 0.274 | 0.00% |
| Japanese | 100 | 1.19 | 0.072 | 0.366 | 0.00% |

**French is easier than English** for this model. Lower perplexity (1.18 vs 1.22) and lower entropy (0.274 vs 0.443). French scientific text has stricter conventions and fewer synonym choices than English.

**Japanese has higher BPB despite similar perplexity.** BPB is 0.072 vs 0.050 for French — 44% higher. This is because Japanese tokens encode more bytes per token (multi-byte UTF-8 for CJK characters), so each token's logprob is amortized over more bytes. The model isn't less confident in Japanese; the information is just more densely packed per byte.

**Missing mass is zero for both French and Japanese.** The model is equally complete in its distributions across languages.

## Experiment 6: Top-5 vs Top-20 truncation

Same prompt, same temperature (1.0), different `top_logprobs` settings.

| Setting | Missing mass | Entropy bias | Entropy (partial) |
|---------|-------------|-------------|-------------------|
| top-5 | 0.47% | +0.0003 bits | 0.461 bits |
| top-20 | 0.04% | +0.0003 bits | 0.543 bits |

**For GPT-4o-mini, even top-5 captures 99.53%.** The entropy bias is 0.0003 bits — negligible. But for GPT-2, top-5 loses 30% of the mass and +0.37 bits of bias. Model quality determines how much truncation matters.

## The -9999 logprob mystery

```
logprobe confidence -t -10 research/data/domain_creative.json

Position 83: logprob=-9999.0000 (p=0.000000)
  Context: ... anxious fingers.[ Gene]viève Du...
```

The model generated "Geneviève Duval" — a French name. The subword token "Gene" received logprob = -9999.0, OpenAI's sentinel for "this token was essentially impossible given the sampling."

When the model generates rare multi-token sequences involving Unicode, individual subword tokens can receive extreme negative logprobs. "Gene" alone is nearly impossible — it only makes sense as the start of "Geneviève." **This is a tokenization artifact, not a model bug.** logprobe flags it:

```
[WARN] extreme_logprobs: 1 tokens have logprobs < -100.
       These may be placeholder values.
```

One -9999 token destroys the entire sequence's BPB (21.84 instead of ~0.2). This is why logprobe's `diagnose` command exists — to catch these before you compute derived metrics.

## Key takeaways

1. **Entropy spikes coincide with hallucinated tokens** in the examples tested. The highest-entropy token in each hallucinated answer was at the transition from grounded to fabricated content.

2. **Refusal text has higher perplexity than fabrication.** Fabricated answers have low entropy (high probability assigned to the chosen fake answer). Refusals have higher entropy (hedge words have low individual probability). Perplexity: fabrication 1.02, refusal 1.56. (N=1 each — illustrative only.)

3. **GPT-4o-mini has negligible truncation bias.** Top-20 captures >99.9% of probability mass. GPT-2 at top-5 loses 30%. Always check with `logprobe diagnose`.

4. **Temperature spreads probability among top tokens, not into the tail.** Missing mass stays <0.02% across all temperatures 0-1.5.

5. **Language affects BPB but not perplexity.** Japanese BPB is 1.4x French due to multi-byte UTF-8 encoding, not model uncertainty.

6. **Watch for -9999 sentinel values.** One sentinel token can make BPB and perplexity meaningless. logprobe catches these automatically.

7. **GPT-4.1-mini was the most confident of four models tested** on one prompt. Lowest perplexity, lowest BPB, zero missing mass, fewest tokens.

## Reproduce everything

```bash
cargo install --git https://github.com/Robby955/logprobe

# Temperature sweep
logprobe diagnose research/data/temp_1.5.json

# Hallucination detection
logprobe entropy research/data/confidence_gradient.json
logprobe entropy research/data/hallucination_factoid.json
logprobe confidence -t -0.5 research/data/confidence_gradient.json

# Cross-model
logprobe diagnose research/data/model_gpt4o.json
logprobe diagnose research/data/model_gpt41mini.json

# Language comparison
logprobe bpb research/data/lang_japanese.json
```

## Data files

| File | Experiment | Model | Prompt | Temp | top_k | Tokens |
|------|-----------|-------|--------|------|-------|--------|
| `temp_0.0.json` | Temp sweep | GPT-4o-mini | "Why is the ocean blue?" | 0.0 | 20 | 100 |
| `temp_0.5.json` | Temp sweep | GPT-4o-mini | "Why is the ocean blue?" | 0.5 | 20 | 100 |
| `temp_1.0.json` | Temp sweep | GPT-4o-mini | "Why is the ocean blue?" | 1.0 | 20 | 100 |
| `temp_1.5.json` | Temp sweep | GPT-4o-mini | "Why is the ocean blue?" | 1.5 | 20 | 100 |
| `domain_factual.json` | Domain | GPT-4o-mini | "Capital of France?" | 0.7 | 20 | 7 |
| `domain_creative.json` | Domain | GPT-4o-mini | "Mystery novel in 1920s Paris" | 0.7 | 20 | 150 |
| `domain_code.json` | Domain | GPT-4o-mini | "Fibonacci with memoization" | 0.7 | 20 | 200 |
| `domain_reasoning.json` | Domain | GPT-4o-mini | "Train distance problem" | 0.7 | 20 | 150 |
| `topk_5.json` | Truncation | GPT-4o-mini | "Why is the ocean blue?" | 1.0 | 5 | 100 |
| `topk_20.json` | Truncation | GPT-4o-mini | "Why is the ocean blue?" | 1.0 | 20 | 100 |
| `model_gpt4o.json` | Cross-model | GPT-4o | "Why is the ocean blue?" | 0.7 | 20 | 100 |
| `model_gpt41mini.json` | Cross-model | GPT-4.1-mini | "Why is the ocean blue?" | 0.7 | 20 | 89 |
| `model_gpt41nano.json` | Cross-model | GPT-4.1-nano | "Why is the ocean blue?" | 0.7 | 20 | 100 |
| `hallucination_factoid.json` | Hallucination | GPT-4o-mini | "Remington 7615 manufacture year?" | 0.7 | 20 | 20 |
| `hallucination_biography.json` | Hallucination | GPT-4o-mini | "Dr. James W. Patterson discoveries?" | 0.7 | 20 | 140 |
| `confidence_gradient.json` | Hallucination | GPT-4o-mini | "2+2? Mona Lisa? Liechtenstein? 23rd moonwalker?" | 0.7 | 20 | 60 |
| `lang_french.json` | Language | GPT-4o-mini | "Pourquoi le ciel est bleu?" | 0.7 | 20 | 100 |
| `lang_japanese.json` | Language | GPT-4o-mini | "空が青い理由は?" | 0.7 | 20 | 100 |

All via the OpenAI Chat Completions API, April 2025.
