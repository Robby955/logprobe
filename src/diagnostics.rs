use crate::math;
use crate::types::{DiagnoseReport, DiagnosticFinding, LogprobSequence, Severity};

const LOG_MASS_ERROR_THRESHOLD: f64 = 2.0;
const LOG_MASS_WARNING_THRESHOLD: f64 = 0.5;
const ENTROPY_BIAS_THRESHOLD: f64 = 0.5;
const EXTREME_LOGPROB_THRESHOLD: f64 = -100.0;
const MASS_EPSILON: f64 = 1e-6;
const CONSTANT_EPSILON: f64 = 1e-6;

/// Run integrity checks on a logprob sequence.
pub fn validate(seq: &LogprobSequence) -> Vec<DiagnosticFinding> {
    let mut findings = Vec::new();

    if seq.tokens.is_empty() {
        findings.push(DiagnosticFinding {
            severity: Severity::Warning,
            check: "empty_sequence".into(),
            message: "sequence contains no tokens".into(),
            position: None,
        });
        return findings;
    }

    for (i, tok) in seq.tokens.iter().enumerate() {
        // Check logprobs are finite
        if !tok.logprob.is_finite() {
            findings.push(DiagnosticFinding {
                severity: Severity::Error,
                check: "finite_logprob".into(),
                message: format!("token {:?} has non-finite logprob: {}", tok.token, tok.logprob),
                position: Some(i),
            });
        }

        // Check logprobs are <= 0
        if tok.logprob > 0.0 {
            findings.push(DiagnosticFinding {
                severity: Severity::Error,
                check: "nonpositive_logprob".into(),
                message: format!(
                    "token {:?} has positive logprob {} (log-probabilities must be <= 0)",
                    tok.token, tok.logprob
                ),
                position: Some(i),
            });
        }

        if let Some(ref top_k) = tok.top_logprobs {
            // Check top_logprobs are sorted descending
            for w in top_k.windows(2) {
                if w[0].logprob < w[1].logprob {
                    findings.push(DiagnosticFinding {
                        severity: Severity::Warning,
                        check: "sorted_top_logprobs".into(),
                        message: format!(
                            "top_logprobs at position {i} not sorted descending: {:?} ({}) before {:?} ({})",
                            w[0].token, w[0].logprob, w[1].token, w[1].logprob
                        ),
                        position: Some(i),
                    });
                    break;
                }
            }

            // Check for duplicate tokens in top_logprobs
            let mut seen = std::collections::HashSet::new();
            for entry in top_k {
                if !seen.insert(&entry.token) {
                    findings.push(DiagnosticFinding {
                        severity: Severity::Error,
                        check: "duplicate_top_token".into(),
                        message: format!(
                            "duplicate token {:?} in top_logprobs at position {i}",
                            entry.token
                        ),
                        position: Some(i),
                    });
                }
            }

            // Check mass <= 1 + epsilon
            let lps: Vec<f64> = top_k.iter().map(|e| e.logprob).collect();
            let mass = math::observed_mass(&lps);
            if mass > 1.0 + MASS_EPSILON {
                findings.push(DiagnosticFinding {
                    severity: Severity::Error,
                    check: "mass_exceeds_one".into(),
                    message: format!(
                        "top_logprobs mass at position {i} is {mass:.6} (exceeds 1.0)"
                    ),
                    position: Some(i),
                });
            }

            // Check all top_logprobs are finite and <= 0
            for entry in top_k {
                if !entry.logprob.is_finite() {
                    findings.push(DiagnosticFinding {
                        severity: Severity::Error,
                        check: "finite_top_logprob".into(),
                        message: format!(
                            "top_logprob {:?} at position {i} is non-finite: {}",
                            entry.token, entry.logprob
                        ),
                        position: Some(i),
                    });
                }
                if entry.logprob > 0.0 {
                    findings.push(DiagnosticFinding {
                        severity: Severity::Error,
                        check: "nonpositive_top_logprob".into(),
                        message: format!(
                            "top_logprob {:?} at position {i} is positive: {}",
                            entry.token, entry.logprob
                        ),
                        position: Some(i),
                    });
                }
            }
        }

        // Check byte consistency where both bytes and token string are present
        if let Some(ref bytes) = tok.bytes {
            let string_bytes = tok.token.as_bytes();
            // Only warn if they differ and bytes is non-empty
            if !bytes.is_empty() && bytes != string_bytes {
                findings.push(DiagnosticFinding {
                    severity: Severity::Warning,
                    check: "byte_consistency".into(),
                    message: format!(
                        "token {:?} at position {i}: bytes field ({} bytes) differs from UTF-8 encoding ({} bytes)",
                        tok.token,
                        bytes.len(),
                        string_bytes.len()
                    ),
                    position: Some(i),
                });
            }
        }
    }

    if findings.is_empty() {
        findings.push(DiagnosticFinding {
            severity: Severity::Ok,
            check: "all_checks".into(),
            message: format!("all validation checks passed ({} tokens)", seq.tokens.len()),
            position: None,
        });
    }

    findings
}

/// Build a structured diagnose report for the sequence.
pub fn diagnose_report(seq: &LogprobSequence) -> DiagnoseReport {
    let validation_findings = validate(seq);
    let has_bytes = seq.tokens.iter().any(|t| t.bytes.is_some());

    if seq.tokens.is_empty() {
        return DiagnoseReport {
            normalization_status: Severity::Warning,
            mean_log_mass: 0.0,
            max_log_mass: 0.0,
            mean_missing_mass: 0.0,
            high_missing_mass_count: 0,
            total_positions: 0,
            entropy_partial: 0.0,
            entropy_normalized: 0.0,
            entropy_bias: 0.0,
            has_bytes,
            findings: validation_findings,
            suspicious_patterns: Vec::new(),
        };
    }

    // Collect positions with top_logprobs for distribution analysis
    let positions_with_topk: Vec<(usize, Vec<f64>)> = seq
        .tokens
        .iter()
        .enumerate()
        .filter_map(|(i, tok)| {
            tok.top_logprobs.as_ref().and_then(|tk| {
                let lps: Vec<f64> = tk.iter().map(|e| e.logprob).collect();
                if lps.is_empty() { None } else { Some((i, lps)) }
            })
        })
        .collect();

    let total_positions = positions_with_topk.len();

    // Log mass estimation
    let (mean_log_mass, max_log_mass) = if positions_with_topk.is_empty() {
        (0.0, 0.0)
    } else {
        let log_masses: Vec<f64> = positions_with_topk
            .iter()
            .map(|(_, lps)| math::estimate_log_mass(lps))
            .collect();
        let mean = log_masses.iter().sum::<f64>() / log_masses.len() as f64;
        let max = log_masses
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        (mean, max)
    };

    let normalization_status = if positions_with_topk.is_empty() {
        Severity::Warning
    } else if mean_log_mass > LOG_MASS_ERROR_THRESHOLD {
        Severity::Error
    } else if mean_log_mass > LOG_MASS_WARNING_THRESHOLD {
        Severity::Warning
    } else {
        Severity::Ok
    };

    // Missing mass analysis
    let (mean_missing_mass, high_missing_mass_count) = if positions_with_topk.is_empty() {
        (0.0, 0)
    } else {
        let missing_masses: Vec<f64> = positions_with_topk
            .iter()
            .map(|(_, lps)| math::missing_mass(lps))
            .collect();
        let mean = missing_masses.iter().sum::<f64>() / missing_masses.len() as f64;
        let high_count = missing_masses.iter().filter(|&&m| m > 0.5).count();
        (mean, high_count)
    };

    // Entropy analysis
    let (entropy_partial, entropy_normalized, entropy_bias) = if positions_with_topk.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        let partials: Vec<f64> = positions_with_topk
            .iter()
            .map(|(_, lps)| math::entropy_bits_partial(lps))
            .collect();
        let normals: Vec<f64> = positions_with_topk
            .iter()
            .map(|(_, lps)| math::entropy_bits_normalized(lps))
            .collect();
        let mean_p = partials.iter().sum::<f64>() / partials.len() as f64;
        let mean_n = normals.iter().sum::<f64>() / normals.len() as f64;
        (mean_p, mean_n, mean_n - mean_p)
    };

    // Suspicious patterns
    let mut suspicious_patterns = Vec::new();
    let all_logprobs: Vec<f64> = seq.tokens.iter().map(|t| t.logprob).collect();

    // Constant logprobs
    if all_logprobs.len() > 1 {
        let first = all_logprobs[0];
        if all_logprobs
            .iter()
            .all(|&lp| (lp - first).abs() < CONSTANT_EPSILON)
        {
            suspicious_patterns.push(DiagnosticFinding {
                severity: Severity::Warning,
                check: "constant_logprobs".into(),
                message: format!(
                    "all {} token logprobs are identical ({first:.6}) — this is suspicious",
                    all_logprobs.len()
                ),
                position: None,
            });
        }
    }

    // All-zero logprobs
    if all_logprobs.iter().all(|&lp| lp == 0.0) {
        suspicious_patterns.push(DiagnosticFinding {
            severity: Severity::Error,
            check: "all_zero_logprobs".into(),
            message: "all logprobs are exactly 0.0 — every token has probability 1, \
                      which is impossible for a real distribution"
                .into(),
            position: None,
        });
    }

    // Extreme values
    let extreme_count = all_logprobs
        .iter()
        .filter(|&&lp| lp < EXTREME_LOGPROB_THRESHOLD)
        .count();
    if extreme_count > 0 {
        suspicious_patterns.push(DiagnosticFinding {
            severity: Severity::Warning,
            check: "extreme_logprobs".into(),
            message: format!(
                "{extreme_count} tokens have logprobs < {} \
                 (probability < 10^-43). These may be placeholder values.",
                EXTREME_LOGPROB_THRESHOLD
            ),
            position: None,
        });
    }

    DiagnoseReport {
        normalization_status,
        mean_log_mass,
        max_log_mass,
        mean_missing_mass,
        high_missing_mass_count,
        total_positions,
        entropy_partial,
        entropy_normalized,
        entropy_bias,
        has_bytes,
        findings: validation_findings,
        suspicious_patterns,
    }
}

/// Run normalization diagnostics — the killer feature.
/// Returns flat findings list for backward compatibility.
pub fn diagnose(seq: &LogprobSequence) -> Vec<DiagnosticFinding> {
    let report = diagnose_report(seq);
    let mut findings = report.findings;

    if seq.tokens.is_empty() {
        return findings;
    }

    // Normalization finding
    match report.normalization_status {
        Severity::Error => {
            findings.push(DiagnosticFinding {
                severity: Severity::Error,
                check: "normalization".into(),
                message: format!(
                    "scores appear UNNORMALIZED (logits, not log-probabilities). \
                     Mean log mass = {:.4}, max log mass = {:.4}. \
                     Perplexity and entropy computations will be incorrect.",
                    report.mean_log_mass, report.max_log_mass
                ),
                position: None,
            });
        }
        Severity::Warning if report.total_positions > 0 => {
            findings.push(DiagnosticFinding {
                severity: Severity::Warning,
                check: "normalization".into(),
                message: format!(
                    "possible normalization issue. Mean log mass = {:.4} \
                     (expected <= 0 for log-probabilities, >> 0 indicates logits)",
                    report.mean_log_mass
                ),
                position: None,
            });
        }
        Severity::Ok => {
            findings.push(DiagnosticFinding {
                severity: Severity::Ok,
                check: "normalization".into(),
                message: format!(
                    "scores appear normalized (mean log mass = {:.4})",
                    report.mean_log_mass
                ),
                position: None,
            });
        }
        _ => {
            // Warning with no positions = no top_logprobs available
            findings.push(DiagnosticFinding {
                severity: Severity::Warning,
                check: "no_top_logprobs".into(),
                message:
                    "no top_logprobs available — normalization diagnostics require top-k data"
                        .into(),
                position: None,
            });
        }
    }

    // Missing mass finding
    if report.total_positions > 0 {
        if report.high_missing_mass_count > 0 {
            findings.push(DiagnosticFinding {
                severity: Severity::Warning,
                check: "missing_mass".into(),
                message: format!(
                    "{}/{} positions have >50% missing probability mass \
                     (mean missing: {:.4}). Entropy estimates at these positions are unreliable.",
                    report.high_missing_mass_count,
                    report.total_positions,
                    report.mean_missing_mass
                ),
                position: None,
            });
        } else {
            findings.push(DiagnosticFinding {
                severity: Severity::Ok,
                check: "missing_mass".into(),
                message: format!(
                    "mean missing mass = {:.4} across {} positions",
                    report.mean_missing_mass, report.total_positions
                ),
                position: None,
            });
        }

        // Entropy bias
        if report.entropy_bias.abs() > ENTROPY_BIAS_THRESHOLD {
            findings.push(DiagnosticFinding {
                severity: Severity::Warning,
                check: "entropy_bias".into(),
                message: format!(
                    "entropy bias from renormalization: {:+.4} bits \
                     (partial: {:.4}, normalized: {:.4}). \
                     Large gap suggests significant truncation.",
                    report.entropy_bias, report.entropy_partial, report.entropy_normalized
                ),
                position: None,
            });
        }
    }

    // Append suspicious patterns
    findings.extend(report.suspicious_patterns);

    findings
}
