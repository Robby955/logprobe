use serde::{Deserialize, Serialize};

use crate::math;
use crate::types::{CompareEntry, CompareReport, LogprobSequence, SequenceSummary, TokenEntropy};

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
        assumed_normalized: false, // run `diagnose` to verify normalization
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

/// Compare two logprob sequences and produce a side-by-side report.
pub fn compute_compare(
    seq_a: &LogprobSequence,
    seq_b: &LogprobSequence,
    label_a: &str,
    label_b: &str,
) -> CompareReport {
    let summary_a = compute_summary(seq_a);
    let summary_b = compute_summary(seq_b);

    let entropy_a = compute_entropy(seq_a);
    let entropy_b = compute_entropy(seq_b);

    let mean_entropy_a = mean_of(entropy_a.iter().map(|e| e.entropy_partial));
    let mean_entropy_b = mean_of(entropy_b.iter().map(|e| e.entropy_partial));

    let bpb_a = match compute_bpb(seq_a) {
        BpbResult::Value { bpb } => Some(bpb),
        BpbResult::Unavailable { .. } => None,
    };
    let bpb_b = match compute_bpb(seq_b) {
        BpbResult::Value { bpb } => Some(bpb),
        BpbResult::Unavailable { .. } => None,
    };

    let delta_missing_mass = match (summary_a.mean_missing_mass, summary_b.mean_missing_mass) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };

    let delta_bpb = match (bpb_a, bpb_b) {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };

    CompareReport {
        file_a: CompareEntry {
            label: label_a.to_string(),
            model: seq_a.model.clone(),
            token_count: summary_a.token_count,
            perplexity: summary_a.perplexity,
            mean_logprob: summary_a.mean_logprob,
            mean_entropy_partial: mean_entropy_a,
            mean_missing_mass: summary_a.mean_missing_mass,
            bpb: bpb_a,
        },
        file_b: CompareEntry {
            label: label_b.to_string(),
            model: seq_b.model.clone(),
            token_count: summary_b.token_count,
            perplexity: summary_b.perplexity,
            mean_logprob: summary_b.mean_logprob,
            mean_entropy_partial: mean_entropy_b,
            mean_missing_mass: summary_b.mean_missing_mass,
            bpb: bpb_b,
        },
        delta_perplexity: summary_b.perplexity - summary_a.perplexity,
        delta_mean_logprob: summary_b.mean_logprob - summary_a.mean_logprob,
        delta_entropy_partial: mean_entropy_b - mean_entropy_a,
        delta_missing_mass,
        delta_bpb,
    }
}

fn mean_of(iter: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for v in iter {
        sum += v;
        count += 1;
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
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
