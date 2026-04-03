use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::io::{self, Read};

use crate::types::{InputFormat, LogprobSequence, TokenLogprob, TopKEntry};

/// Parse logprobs from a reader, auto-detecting format unless overridden.
pub fn parse_input(
    reader: impl Read,
    format_override: Option<InputFormat>,
    strict: bool,
) -> Result<LogprobSequence> {
    let mut buf = String::new();
    let mut reader = io::BufReader::new(reader);
    reader.read_to_string(&mut buf)?;
    parse_string(&buf, format_override, strict)
}

/// Parse logprobs from a string, auto-detecting format unless overridden.
pub fn parse_string(
    input: &str,
    format_override: Option<InputFormat>,
    strict: bool,
) -> Result<LogprobSequence> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("empty input");
    }

    let format = match format_override {
        Some(f) => f,
        None => detect_format(trimmed, strict)?,
    };

    match format {
        InputFormat::OpenAI => parse_openai(trimmed),
        InputFormat::VllmFlat => parse_vllm(trimmed),
        InputFormat::JsonlStream => parse_jsonl(trimmed),
    }
}

/// Detect the input format from content structure.
fn detect_format(input: &str, strict: bool) -> Result<InputFormat> {
    // Try parsing as JSON first
    if let Ok(val) = serde_json::from_str::<Value>(input) {
        return detect_format_json(&val, strict);
    }

    // Check if it looks like JSONL (multiple lines, each valid JSON)
    let lines: Vec<&str> = input.lines().filter(|l| !l.trim().is_empty()).collect();
    if !lines.is_empty() && lines.iter().all(|l| serde_json::from_str::<Value>(l).is_ok()) {
        return Ok(InputFormat::JsonlStream);
    }

    bail!("could not detect input format: not valid JSON or JSONL")
}

fn detect_format_json(val: &Value, strict: bool) -> Result<InputFormat> {
    // OpenAI: choices[0].logprobs.content is an array of objects with "token" + "logprob"
    if let Some(content) = val
        .pointer("/choices/0/logprobs/content")
        .and_then(|v| v.as_array())
        && content
            .first()
            .map(|c| c.get("token").is_some() && c.get("logprob").is_some())
            .unwrap_or(false)
    {
        return Ok(InputFormat::OpenAI);
    }

    // vLLM/Together: choices[0].logprobs has "tokens" array and "token_logprobs" array
    if let Some(logprobs) = val.pointer("/choices/0/logprobs")
        && logprobs.get("tokens").and_then(|v| v.as_array()).is_some()
        && logprobs
            .get("token_logprobs")
            .and_then(|v| v.as_array())
            .is_some()
    {
        return Ok(InputFormat::VllmFlat);
    }

    if strict {
        bail!("--strict-format: could not unambiguously detect format from JSON structure")
    }

    // Fallback: if it's an array of objects with token+logprob, treat as JSONL-style
    if let Some(arr) = val.as_array()
        && arr
            .first()
            .map(|v| v.get("token").is_some() && v.get("logprob").is_some())
            .unwrap_or(false)
    {
        return Ok(InputFormat::JsonlStream);
    }

    bail!("could not detect format from JSON structure")
}

/// Parse OpenAI Chat Completions format.
fn parse_openai(input: &str) -> Result<LogprobSequence> {
    let val: Value = serde_json::from_str(input).context("invalid JSON")?;
    let model = val.get("model").and_then(|m| m.as_str()).map(String::from);

    let content = val
        .pointer("/choices/0/logprobs/content")
        .and_then(|v| v.as_array())
        .context("missing choices[0].logprobs.content")?;

    let mut tokens = Vec::with_capacity(content.len());
    let mut total_logprob = 0.0;

    for item in content {
        let token = item
            .get("token")
            .and_then(|v| v.as_str())
            .context("missing token field")?
            .to_string();
        let logprob = item
            .get("logprob")
            .and_then(|v| v.as_f64())
            .context("missing logprob field")?;

        let bytes = item.get("bytes").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|b| b.as_u64().map(|n| n as u8))
                .collect()
        });

        let top_logprobs = item
            .get("top_logprobs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|entry| {
                        let t = entry.get("token")?.as_str()?.to_string();
                        let lp = entry.get("logprob")?.as_f64()?;
                        Some(TopKEntry {
                            token: t,
                            logprob: lp,
                        })
                    })
                    .collect()
            });

        total_logprob += logprob;
        tokens.push(TokenLogprob {
            token,
            logprob,
            bytes,
            top_logprobs,
        });
    }

    Ok(LogprobSequence {
        tokens,
        model,
        format_detected: InputFormat::OpenAI.to_string(),
        total_logprob,
        is_normalized: None,
    })
}

/// Parse vLLM / Together flat format.
fn parse_vllm(input: &str) -> Result<LogprobSequence> {
    let val: Value = serde_json::from_str(input).context("invalid JSON")?;
    let model = val.get("model").and_then(|m| m.as_str()).map(String::from);

    let logprobs_obj = val
        .pointer("/choices/0/logprobs")
        .context("missing choices[0].logprobs")?;

    let token_strs = logprobs_obj
        .get("tokens")
        .and_then(|v| v.as_array())
        .context("missing tokens array")?;

    let token_lps = logprobs_obj
        .get("token_logprobs")
        .and_then(|v| v.as_array())
        .context("missing token_logprobs array")?;

    let top_lps = logprobs_obj
        .get("top_logprobs")
        .and_then(|v| v.as_array());

    if token_strs.len() != token_lps.len() {
        bail!("tokens and token_logprobs arrays have different lengths");
    }

    let mut tokens = Vec::with_capacity(token_strs.len());
    let mut total_logprob = 0.0;

    for (i, (ts, tl)) in token_strs.iter().zip(token_lps.iter()).enumerate() {
        let token = ts
            .as_str()
            .context(format!("token at index {i} is not a string"))?
            .to_string();
        let logprob = tl
            .as_f64()
            .context(format!("logprob at index {i} is not a number"))?;

        let top_logprobs = top_lps.and_then(|arr| {
            arr.get(i)
                .and_then(|v| v.as_object())
                .map(|obj| {
                    let mut entries: Vec<TopKEntry> = obj
                        .iter()
                        .map(|(k, v)| TopKEntry {
                            token: k.clone(),
                            logprob: v.as_f64().unwrap_or(f64::NEG_INFINITY),
                        })
                        .collect();
                    // JSON objects have arbitrary key order — sort descending
                    entries.sort_by(|a, b| b.logprob.partial_cmp(&a.logprob).unwrap_or(std::cmp::Ordering::Equal));
                    entries
                })
        });

        total_logprob += logprob;
        tokens.push(TokenLogprob {
            token,
            logprob,
            bytes: None,
            top_logprobs,
        });
    }

    Ok(LogprobSequence {
        tokens,
        model,
        format_detected: InputFormat::VllmFlat.to_string(),
        total_logprob,
        is_normalized: None,
    })
}

/// Parse JSONL stream format (one {token, logprob} per line).
fn parse_jsonl(input: &str) -> Result<LogprobSequence> {
    let mut tokens = Vec::new();
    let mut total_logprob = 0.0;

    // Handle both actual JSONL and a JSON array
    let items: Vec<Value> = if input.trim_start().starts_with('[') {
        serde_json::from_str(input).context("invalid JSON array")?
    } else {
        input
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str)
            .collect::<Result<Vec<_>, _>>()
            .context("invalid JSONL")?
    };

    for item in &items {
        let token = item
            .get("token")
            .and_then(|v| v.as_str())
            .context("missing token field in JSONL entry")?
            .to_string();
        let logprob = item
            .get("logprob")
            .and_then(|v| v.as_f64())
            .context("missing logprob field in JSONL entry")?;

        let bytes = item.get("bytes").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|b| b.as_u64().map(|n| n as u8))
                .collect()
        });

        total_logprob += logprob;
        tokens.push(TokenLogprob {
            token,
            logprob,
            bytes,
            top_logprobs: None,
        });
    }

    Ok(LogprobSequence {
        tokens,
        model: None,
        format_detected: InputFormat::JsonlStream.to_string(),
        total_logprob,
        is_normalized: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_openai() {
        let input = r#"{"choices":[{"logprobs":{"content":[{"token":"Hi","logprob":-0.5}]}}]}"#;
        let seq = parse_string(input, None, false).unwrap();
        assert_eq!(seq.format_detected, "openai");
        assert_eq!(seq.tokens.len(), 1);
    }

    #[test]
    fn test_detect_vllm() {
        let input = r#"{"choices":[{"logprobs":{"tokens":["Hi"],"token_logprobs":[-0.5]}}]}"#;
        let seq = parse_string(input, None, false).unwrap();
        assert_eq!(seq.format_detected, "vllm");
    }

    #[test]
    fn test_detect_jsonl() {
        let input = "{\"token\":\"Hi\",\"logprob\":-0.5}\n{\"token\":\" there\",\"logprob\":-1.0}";
        let seq = parse_string(input, None, false).unwrap();
        assert_eq!(seq.format_detected, "jsonl");
        assert_eq!(seq.tokens.len(), 2);
    }

    #[test]
    fn test_strict_rejects_ambiguous() {
        let input = r#"[{"token":"Hi","logprob":-0.5}]"#;
        // This is valid as jsonl-style array — strict should still parse it
        // since it falls through to the array detection
        let result = parse_string(input, None, true);
        // strict mode rejects if JSON structure doesn't match openai/vllm
        assert!(result.is_err());
    }
}
