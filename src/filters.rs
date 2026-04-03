use crate::types::{LogprobSequence, LowConfidenceToken, TokenEntropy};

const MIN_STD_DEV: f64 = 1e-10;

/// Find tokens with logprob below the given threshold.
pub fn find_low_confidence(
    seq: &LogprobSequence,
    threshold: f64,
    context_size: usize,
) -> Vec<LowConfidenceToken> {
    let mut results = Vec::new();

    for (i, tok) in seq.tokens.iter().enumerate() {
        if tok.logprob < threshold {
            let context_before: Vec<String> = seq.tokens
                [i.saturating_sub(context_size)..i]
                .iter()
                .map(|t| t.token.clone())
                .collect();

            let end = (i + 1 + context_size).min(seq.tokens.len());
            let context_after: Vec<String> = seq.tokens[i + 1..end]
                .iter()
                .map(|t| t.token.clone())
                .collect();

            results.push(LowConfidenceToken {
                position: i,
                token: tok.token.clone(),
                logprob: tok.logprob,
                probability: tok.logprob.exp(),
                context_before,
                context_after,
            });
        }
    }

    results
}

/// Detect entropy spikes — positions where entropy is unusually high.
/// Returns indices of positions where entropy exceeds mean + `std_devs` standard deviations.
pub fn detect_entropy_spikes(entropies: &[TokenEntropy], std_devs: f64) -> Vec<usize> {
    if entropies.is_empty() {
        return Vec::new();
    }

    let values: Vec<f64> = entropies.iter().map(|e| e.entropy_normalized).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();

    if std_dev < MIN_STD_DEV {
        return Vec::new();
    }

    let spike_threshold = mean + std_devs * std_dev;

    entropies
        .iter()
        .enumerate()
        .filter(|(_, e)| e.entropy_normalized > spike_threshold)
        .map(|(i, _)| i)
        .collect()
}
