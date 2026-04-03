use serde::{Deserialize, Serialize};

use crate::math;
use crate::types::{LogprobSequence, SequenceSummary, TokenEntropy};

const MISSING_MASS_UNRELIABILITY_THRESHOLD: f64 = 0.5;

/// Compute overall summary statistics for a logprob sequence.
pub fn compute_summary(seq: &LogprobSequence) -> SequenceSummary {
    let logprobs: Vec<f64> = seq.tokens.iter().map(|t| t.logprob).collect();

    let mean_missing_mass = compute_mean_missing_mass(seq);

    SequenceSummary {
        token_count: seq.tokens.len(),
        mean_logprob: math::mean_logprob(&logprobs),
        total_logprob: seq.total_logprob,
        perplexity: math::perplexity(&logprobs),
        assumed_normalized: seq.is_normalized.unwrap_or(false),
        mean_missing_mass,
    }
}

/// Compute per-token entropy metrics.
pub fn compute_entropy(seq: &LogprobSequence) -> Vec<TokenEntropy> {
    seq.tokens
        .iter()
        .enumerate()
        .map(|(i, tok)| {
            let (partial, normalized, mm) = match &tok.top_logprobs {
                Some(top_k) if !top_k.is_empty() => {
                    let lps: Vec<f64> = top_k.iter().map(|e| e.logprob).collect();
                    (
                        math::entropy_bits_partial(&lps),
                        math::entropy_bits_normalized(&lps),
                        math::missing_mass(&lps),
                    )
                }
                _ => (0.0, 0.0, 1.0), // No top-k: all mass is missing
            };

            TokenEntropy {
                position: i,
                token: tok.token.clone(),
                entropy_partial: partial,
                entropy_normalized: normalized,
                missing_mass: mm,
                unreliable: mm > MISSING_MASS_UNRELIABILITY_THRESHOLD,
            }
        })
        .collect()
}

/// Result of BPB computation — either a value or an error explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum BpbResult {
    #[serde(rename = "ok")]
    Value { bpb: f64 },
    #[serde(rename = "error")]
    Unavailable { reason: String },
}

/// Compute bits-per-byte, refusing if byte counts are unavailable.
pub fn compute_bpb(seq: &LogprobSequence) -> BpbResult {
    let logprobs: Vec<f64> = seq.tokens.iter().map(|t| t.logprob).collect();
    let byte_counts: Vec<Option<usize>> = seq
        .tokens
        .iter()
        .map(|t| t.bytes.as_ref().map(|b| b.len()))
        .collect();

    let has_any_bytes = byte_counts.iter().any(|b| b.is_some());
    if !has_any_bytes {
        return BpbResult::Unavailable {
            reason: "no byte counts available in logprob data. \
             BPB requires explicit byte arrays from the API — \
             logprobe refuses to fallback to token.as_bytes().len() \
             because BPE tokens have leading spaces and special encodings \
             that make UTF-8 byte length incorrect."
                .into(),
        };
    }

    let missing: Vec<usize> = byte_counts
        .iter()
        .enumerate()
        .filter_map(|(i, b)| if b.is_none() { Some(i) } else { None })
        .collect();

    if !missing.is_empty() {
        return BpbResult::Unavailable {
            reason: format!(
                "{} out of {} tokens are missing byte counts (first at position {}). \
                 BPB requires byte counts for ALL tokens.",
                missing.len(),
                seq.tokens.len(),
                missing[0]
            ),
        };
    }

    match math::bpb(&logprobs, &byte_counts) {
        Some(val) => BpbResult::Value { bpb: val },
        None => BpbResult::Unavailable {
            reason: "BPB computation failed (zero total bytes?)".into(),
        },
    }
}

fn compute_mean_missing_mass(seq: &LogprobSequence) -> Option<f64> {
    let masses: Vec<f64> = seq
        .tokens
        .iter()
        .filter_map(|tok| {
            tok.top_logprobs.as_ref().map(|top_k| {
                let lps: Vec<f64> = top_k.iter().map(|e| e.logprob).collect();
                math::missing_mass(&lps)
            })
        })
        .collect();

    if masses.is_empty() {
        None
    } else {
        Some(masses.iter().sum::<f64>() / masses.len() as f64)
    }
}
