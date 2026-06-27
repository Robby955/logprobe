/// Compute the sum of probabilities from logprobs (the observed mass).
/// Returns a value in `[0, 1]` for properly normalized distributions.
#[must_use]
pub fn observed_mass(logprobs: &[f64]) -> f64 {
    logprobs.iter().map(|lp| lp.exp()).sum()
}

/// Compute missing probability mass from top-k logprobs.
///
/// `missing_mass = 1 - sum(exp(logprob_i))`, clamped to `[0, 1]` to absorb
/// floating-point imprecision.
///
/// # Examples
///
/// ```
/// use logprobe::math::missing_mass;
/// // A single observed token with probability 0.5 leaves half the mass unseen.
/// let lps = [0.5_f64.ln()];
/// assert!((missing_mass(&lps) - 0.5).abs() < 1e-12);
/// ```
#[must_use]
pub fn missing_mass(logprobs: &[f64]) -> f64 {
    (1.0 - observed_mass(logprobs)).clamp(0.0, 1.0)
}

/// Sum of surprisals (in bits) over the observed top-k, weighted by each
/// observed probability.
///
/// This is not a true entropy: the observed probabilities do not sum to 1.
/// Because every omitted token contributes a non-negative term
/// `-p log2 p >= 0`, this quantity is an unconditional lower bound on the true
/// Shannon entropy of the full distribution.
#[must_use]
pub fn entropy_bits_partial(logprobs: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &lp in logprobs {
        let p = lp.exp();
        if p > 0.0 {
            // H = -sum(p * log2(p)) = -sum(p * lp / ln2)
            entropy -= p * lp / std::f64::consts::LN_2;
        }
    }
    entropy
}

/// Entropy (in bits) of the observed top-k conditional distribution, after
/// renormalizing it to sum to 1.
///
/// Its bias relative to the true Shannon entropy of the full distribution has
/// indeterminate sign: renormalizing can make the observed slice look either
/// more or less uniform than the full distribution, so this can sit above or
/// below the true value depending on the data. It is still useful for
/// comparing positions that expose different top-k sizes.
#[must_use]
pub fn entropy_bits_normalized(logprobs: &[f64]) -> f64 {
    let mass = observed_mass(logprobs);
    if mass <= 0.0 {
        return 0.0;
    }
    let mut entropy = 0.0;
    for &lp in logprobs {
        let p = lp.exp() / mass;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Mean log-probability across a sequence of token logprobs.
#[must_use]
pub fn mean_logprob(logprobs: &[f64]) -> f64 {
    if logprobs.is_empty() {
        return 0.0;
    }
    logprobs.iter().sum::<f64>() / logprobs.len() as f64
}

/// Perplexity = `exp(-mean_logprob)`.
/// Only valid when logprobs are from a normalized distribution.
#[must_use]
pub fn perplexity(logprobs: &[f64]) -> f64 {
    (-mean_logprob(logprobs)).exp()
}

/// Bits per byte (BPB).
///
/// Requires explicit byte counts and refuses to guess from token-string
/// encoding. Returns `None` if any byte count is missing or zero, or if the
/// slice lengths differ.
#[must_use]
pub fn bpb(logprobs: &[f64], byte_counts: &[Option<usize>]) -> Option<f64> {
    if logprobs.len() != byte_counts.len() {
        return None;
    }
    let mut total_logprob = 0.0;
    let mut total_bytes: usize = 0;
    for (lp, bc) in logprobs.iter().zip(byte_counts.iter()) {
        match bc {
            Some(n) if *n > 0 => {
                total_logprob += lp;
                total_bytes += n;
            }
            _ => return None,
        }
    }
    if total_bytes == 0 {
        return None;
    }
    // BPB = -total_logprob / (total_bytes * ln(2))
    Some(-total_logprob / (total_bytes as f64 * std::f64::consts::LN_2))
}

/// Estimate the log observed mass from a set of scores.
///
/// `log_mass = log(sum(exp(scores)))`.
/// - ≈ 0 for complete normalized distributions (all tokens present).
/// - < 0 for truncated top-k log-probabilities (expected — mass is missing).
/// - ≫ 0 for unnormalized logits (scores were never softmaxed).
///
/// Uses the log-sum-exp trick for numerical stability.
#[must_use]
pub fn estimate_log_mass(logprobs: &[f64]) -> f64 {
    if logprobs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_lp = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_lp == f64::NEG_INFINITY {
        return f64::NEG_INFINITY; // all -inf → invalid distribution, not "normalized"
    }
    if max_lp == f64::INFINITY {
        return f64::INFINITY; // +inf logprobs → raw logits or corrupt data
    }
    let sum_exp: f64 = logprobs.iter().map(|lp| (lp - max_lp).exp()).sum();
    max_lp + sum_exp.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_mass_full_distribution() {
        // Two tokens with probabilities 0.7 and 0.3 → missing mass ≈ 0
        let lps = [0.7_f64.ln(), 0.3_f64.ln()];
        let mm = missing_mass(&lps);
        assert!(mm.abs() < 1e-10, "expected ~0, got {mm}");
    }

    #[test]
    fn test_missing_mass_truncated() {
        // Single token with probability 0.5 → missing mass = 0.5
        let lps = [0.5_f64.ln()];
        let mm = missing_mass(&lps);
        assert!((mm - 0.5).abs() < 1e-10, "expected ~0.5, got {mm}");
    }

    #[test]
    fn test_entropy_uniform_two() {
        // Uniform distribution over 2 tokens → 1 bit
        let lps = [0.5_f64.ln(), 0.5_f64.ln()];
        let h = entropy_bits_partial(&lps);
        assert!((h - 1.0).abs() < 1e-10, "expected 1 bit, got {h}");
    }

    #[test]
    fn test_entropy_normalized_equals_partial_when_complete() {
        let lps = [0.7_f64.ln(), 0.3_f64.ln()];
        let hp = entropy_bits_partial(&lps);
        let hn = entropy_bits_normalized(&lps);
        assert!(
            (hp - hn).abs() < 1e-10,
            "should be equal for complete distribution: {hp} vs {hn}"
        );
    }

    #[test]
    fn test_perplexity_certain() {
        // All logprobs = 0 (probability 1) → perplexity 1
        let lps = [0.0, 0.0, 0.0];
        assert!((perplexity(&lps) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bpb_requires_bytes() {
        let lps = [-1.0, -2.0];
        let bc = [None, Some(3)];
        assert!(bpb(&lps, &bc).is_none());
    }

    #[test]
    fn test_bpb_correct() {
        let lps = [-1.0];
        let bc = [Some(4)];
        let result = bpb(&lps, &bc).unwrap();
        let expected = 1.0 / (4.0 * std::f64::consts::LN_2);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_mass_normalized() {
        // Probabilities 0.7, 0.3 → log mass ≈ 0
        let lps = [0.7_f64.ln(), 0.3_f64.ln()];
        let lm = estimate_log_mass(&lps);
        assert!(lm.abs() < 1e-10, "expected ~0 for normalized, got {lm}");
    }

    #[test]
    fn test_log_mass_unnormalized() {
        // Raw logits like [2.0, 1.0, 0.5] → log mass >> 0
        let logits = [2.0, 1.0, 0.5];
        let lm = estimate_log_mass(&logits);
        assert!(lm > 1.0, "expected large log mass for logits, got {lm}");
    }
}
