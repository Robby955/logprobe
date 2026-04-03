#!/usr/bin/env python3
"""
Generate real logprob data from GPT-2 for logprobe demonstration.

Runs GPT-2 (124M) locally via HuggingFace transformers — no API key needed.
Outputs logprob data in OpenAI, vLLM, or JSONL format that logprobe can analyze.

Usage:
    # Generate normalized logprobs (OpenAI format)
    python generate_logprobs.py --format openai --output gpt2_openai.json

    # Generate raw logits (unnormalized — logprobe should catch this)
    python generate_logprobs.py --format openai --raw-logits --output gpt2_logits_openai.json

    # Generate in vLLM format
    python generate_logprobs.py --format vllm --output gpt2_vllm.json

    # Generate JSONL stream
    python generate_logprobs.py --format jsonl --output gpt2_stream.jsonl

    # Custom prompt and generation length
    python generate_logprobs.py --prompt "Once upon a time" --max-tokens 30

    # Teacher-forced mode: score an existing text instead of generating
    python generate_logprobs.py --score "The capital of France is Paris."

Requirements:
    pip install torch transformers
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(device: str = "cpu"):
    """Load GPT-2 (124M) — small enough to run on any machine."""
    print("Loading GPT-2...", file=sys.stderr)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    model.eval()
    print("Model ready.", file=sys.stderr)
    return model, tokenizer


def score_text(model, tokenizer, text: str, top_k: int = 5, raw_logits: bool = False, device: str = "cpu"):
    """
    Teacher-forced scoring: compute logprobs for each token in a fixed text.
    Returns a list of token records with logprob, bytes, and top_logprobs.
    """
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    n_tokens = input_ids.shape[1]

    if n_tokens < 2:
        raise ValueError("Text must encode to at least 2 tokens")

    records = []

    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        all_logits = outputs.logits[0]  # (seq_len, vocab_size)

    # For each position i, logits[i] predicts token at position i+1
    for i in range(n_tokens - 1):
        target_id = input_ids[0, i + 1].item()
        logits_i = all_logits[i]  # (vocab_size,)

        if raw_logits:
            # Output raw logit scores — NOT log-probabilities.
            # logprobe should detect this as unnormalized.
            scores = logits_i
        else:
            # Proper log-softmax → log-probabilities
            scores = F.log_softmax(logits_i, dim=-1)

        # Get top-k
        topk_values, topk_indices = torch.topk(scores, top_k)

        # Build top_logprobs entries
        top_entries = []
        for val, idx in zip(topk_values, topk_indices):
            tok_str = tokenizer.decode([idx.item()])
            tok_bytes = list(tok_str.encode("utf-8"))
            top_entries.append({
                "token": tok_str,
                "logprob": round(val.item(), 6),
                "bytes": tok_bytes,
            })

        # The actual chosen token
        chosen_token = tokenizer.decode([target_id])
        chosen_score = scores[target_id].item()
        chosen_bytes = list(chosen_token.encode("utf-8"))

        records.append({
            "token": chosen_token,
            "logprob": round(chosen_score, 6),
            "bytes": chosen_bytes,
            "top_logprobs": top_entries,
        })

    return records


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 20,
                  top_k: int = 5, raw_logits: bool = False, device: str = "cpu"):
    """
    Autoregressive generation with logprob extraction.
    Generates token-by-token and records logprobs at each step.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    records = []

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            logits_next = outputs.logits[0, -1, :]  # (vocab_size,)

            if raw_logits:
                scores = logits_next
            else:
                scores = F.log_softmax(logits_next, dim=-1)

            # Sample greedily (argmax) for reproducibility
            chosen_id = torch.argmax(scores).item()
            chosen_score = scores[chosen_id].item()
            chosen_token = tokenizer.decode([chosen_id])
            chosen_bytes = list(chosen_token.encode("utf-8"))

            # Top-k
            topk_values, topk_indices = torch.topk(scores, top_k)
            top_entries = []
            for val, idx in zip(topk_values, topk_indices):
                tok_str = tokenizer.decode([idx.item()])
                tok_bytes = list(tok_str.encode("utf-8"))
                top_entries.append({
                    "token": tok_str,
                    "logprob": round(val.item(), 6),
                    "bytes": tok_bytes,
                })

            records.append({
                "token": chosen_token,
                "logprob": round(chosen_score, 6),
                "bytes": chosen_bytes,
                "top_logprobs": top_entries,
            })

            # Stop on EOS
            if chosen_id == tokenizer.eos_token_id:
                break

            # Append for next step
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[chosen_id]], device=device)
            ], dim=1)

    return records


def format_openai(records: list, model_name: str = "gpt2") -> dict:
    """Format records as an OpenAI Chat Completions API response."""
    content = []
    for r in records:
        entry = {
            "token": r["token"],
            "logprob": r["logprob"],
            "bytes": r["bytes"],
            "top_logprobs": [
                {"token": t["token"], "logprob": t["logprob"], "bytes": t["bytes"]}
                for t in r["top_logprobs"]
            ],
        }
        content.append(entry)

    generated_text = "".join(r["token"] for r in records)

    return {
        "id": "logprobe-demo",
        "object": "chat.completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text,
            },
            "logprobs": {
                "content": content,
            },
        }],
    }


def format_vllm(records: list, model_name: str = "gpt2") -> dict:
    """Format records as a vLLM/Together Completions API response."""
    tokens = [r["token"] for r in records]
    token_logprobs = [r["logprob"] for r in records]

    # vLLM top_logprobs: list of dicts mapping token -> logprob
    top_logprobs = []
    for r in records:
        top_dict = {}
        for t in r["top_logprobs"]:
            top_dict[t["token"]] = t["logprob"]
        top_logprobs.append(top_dict)

    generated_text = "".join(tokens)

    return {
        "id": "logprobe-demo",
        "object": "text_completion",
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": generated_text,
            "logprobs": {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            },
        }],
    }


def format_jsonl(records: list) -> str:
    """Format records as JSONL (one token per line, minimal)."""
    lines = []
    for r in records:
        entry = {
            "token": r["token"],
            "logprob": r["logprob"],
            "bytes": r["bytes"],
        }
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate logprob data from GPT-2 for logprobe demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score existing text (teacher-forced, deterministic)
  python generate_logprobs.py --score "The capital of France is Paris."

  # Generate continuation from prompt
  python generate_logprobs.py --prompt "Once upon a time" --max-tokens 30

  # Output raw logits to demonstrate logprobe catching unnormalized data
  python generate_logprobs.py --score "Hello world" --raw-logits

  # All three formats at once
  python generate_logprobs.py --score "The quick brown fox" --format openai --output demo_openai.json
  python generate_logprobs.py --score "The quick brown fox" --format vllm --output demo_vllm.json
  python generate_logprobs.py --score "The quick brown fox" --format jsonl --output demo_stream.jsonl
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--score", type=str, metavar="TEXT",
        help="Teacher-forced: score each token in the given text (deterministic)"
    )
    mode.add_argument(
        "--prompt", type=str, metavar="TEXT",
        help="Autoregressive: generate a continuation from this prompt"
    )

    parser.add_argument(
        "--format", choices=["openai", "vllm", "jsonl"], default="openai",
        help="Output format (default: openai)"
    )
    parser.add_argument(
        "--raw-logits", action="store_true",
        help="Output raw logit scores instead of log-probabilities. "
             "logprobe should detect this as unnormalized."
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top logprobs to include (default: 5)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=20,
        help="Max tokens to generate in prompt mode (default: 20)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file (default: stdout)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: cpu, cuda, mps (default: cpu)"
    )

    args = parser.parse_args()

    model, tokenizer = load_model(args.device)

    model_label = "gpt2"
    if args.raw_logits:
        model_label = "gpt2 (RAW LOGITS — not log-probabilities)"

    if args.score:
        records = score_text(
            model, tokenizer, args.score,
            top_k=args.top_k, raw_logits=args.raw_logits, device=args.device,
        )
    else:
        records = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens, top_k=args.top_k,
            raw_logits=args.raw_logits, device=args.device,
        )

    # Format output
    if args.format == "openai":
        data = format_openai(records, model_label)
        output = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    elif args.format == "vllm":
        data = format_vllm(records, model_label)
        output = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    elif args.format == "jsonl":
        output = format_jsonl(records)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
