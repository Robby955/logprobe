#!/usr/bin/env python3
"""Verify that pre-generated fixture files have internally consistent values."""

import json
import math
import sys
from pathlib import Path


def check_openai(path: str, expect_normalized: bool = True) -> list[str]:
    errors = []
    data = json.loads(Path(path).read_text())
    content = data["choices"][0]["logprobs"]["content"]

    for i, entry in enumerate(content):
        token = entry["token"]

        # Check bytes match UTF-8 encoding
        expected_bytes = list(token.encode("utf-8"))
        if entry.get("bytes") and entry["bytes"] != expected_bytes:
            errors.append(f"  pos {i} ({token!r}): bytes mismatch: got {entry['bytes']}, expected {expected_bytes}")

        # Check top_logprobs
        if "top_logprobs" in entry:
            # Mass check only for normalized data
            if expect_normalized:
                probs = [math.exp(t["logprob"]) for t in entry["top_logprobs"]]
                total = sum(probs)
                if total > 1.0 + 1e-6:
                    errors.append(f"  pos {i} ({token!r}): top-k mass = {total:.6f} > 1.0")

            # Check sorted descending
            lps = [t["logprob"] for t in entry["top_logprobs"]]
            for j in range(len(lps) - 1):
                if lps[j] < lps[j + 1]:
                    errors.append(f"  pos {i} ({token!r}): top_logprobs not sorted descending")
                    break

            # Check each top_logprob bytes field
            for t in entry["top_logprobs"]:
                if "bytes" in t:
                    exp = list(t["token"].encode("utf-8"))
                    if t["bytes"] != exp:
                        errors.append(f"  pos {i}: top token {t['token']!r} bytes mismatch")

    return errors


def check_normalized(path: str) -> list[str]:
    """Additional checks for normalized logprobs (should all be ≤ 0)."""
    errors = []
    data = json.loads(Path(path).read_text())
    content = data["choices"][0]["logprobs"]["content"]

    for i, entry in enumerate(content):
        if entry["logprob"] > 0:
            errors.append(f"  pos {i} ({entry['token']!r}): logprob {entry['logprob']} > 0 in normalized file")
        if "top_logprobs" in entry:
            for t in entry["top_logprobs"]:
                if t["logprob"] > 0:
                    errors.append(f"  pos {i}: top token {t['token']!r} logprob {t['logprob']} > 0 in normalized file")

    return errors


def check_unnormalized(path: str) -> list[str]:
    """Verify that logits file has positive values (confirming it's unnormalized)."""
    errors = []
    data = json.loads(Path(path).read_text())
    content = data["choices"][0]["logprobs"]["content"]

    positive_count = sum(1 for e in content if e["logprob"] > 0)
    if positive_count == 0:
        errors.append("  logits file has no positive logprobs — are these really raw logits?")

    return errors


def main():
    demo_dir = Path(__file__).parent
    all_ok = True

    # Normalized OpenAI
    print("Checking gpt2_openai.json (normalized)...")
    errs = check_openai(demo_dir / "gpt2_openai.json")
    errs += check_normalized(demo_dir / "gpt2_openai.json")
    if errs:
        all_ok = False
        for e in errs:
            print(e)
    else:
        print("  OK")

    # Logits OpenAI
    print("Checking gpt2_logits_openai.json (raw logits)...")
    errs = check_openai(demo_dir / "gpt2_logits_openai.json", expect_normalized=False)
    errs += check_unnormalized(demo_dir / "gpt2_logits_openai.json")
    if errs:
        all_ok = False
        for e in errs:
            print(e)
    else:
        print("  OK")

    # vLLM
    print("Checking gpt2_vllm.json...")
    vllm_ok = True
    data = json.loads((demo_dir / "gpt2_vllm.json").read_text())
    lp_obj = data["choices"][0]["logprobs"]
    if len(lp_obj["tokens"]) != len(lp_obj["token_logprobs"]):
        print("  tokens/token_logprobs length mismatch")
        vllm_ok = False
    if len(lp_obj["tokens"]) != len(lp_obj["top_logprobs"]):
        print("  tokens/top_logprobs length mismatch")
        vllm_ok = False
    for i, top in enumerate(lp_obj["top_logprobs"]):
        total = sum(math.exp(v) for v in top.values())
        if total > 1.0 + 1e-6:
            print(f"  pos {i}: top-k mass = {total:.6f} > 1.0")
            vllm_ok = False
    if vllm_ok:
        print("  OK")
    else:
        all_ok = False

    # JSONL
    print("Checking gpt2_stream.jsonl...")
    jsonl_ok = True
    lines = (demo_dir / "gpt2_stream.jsonl").read_text().strip().split("\n")
    for j, line in enumerate(lines):
        entry = json.loads(line)
        if entry["logprob"] > 0:
            print(f"  line {j}: positive logprob in JSONL")
            jsonl_ok = False
        if "bytes" in entry:
            exp = list(entry["token"].encode("utf-8"))
            if entry["bytes"] != exp:
                print(f"  line {j}: bytes mismatch for {entry['token']!r}")
                jsonl_ok = False
    if jsonl_ok:
        print("  OK")
    else:
        all_ok = False

    print()
    if all_ok:
        print("All fixtures consistent.")
    else:
        print("ERRORS found — fix before committing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
