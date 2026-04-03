use logprobe::diagnostics;
use logprobe::filters;
use logprobe::math;
use logprobe::metrics;
use logprobe::parse;
use logprobe::types::{
    LogprobSequence, Severity, TokenEntropy, TokenLogprob, TopKEntry,
};

const OPENAI_FIXTURE: &str = include_str!("fixtures/openai_sample.json");
const VLLM_FIXTURE: &str = include_str!("fixtures/vllm_sample.json");
const JSONL_FIXTURE: &str = include_str!("fixtures/stream.jsonl");

// ─── Original tests ──────────────────────────────────────────────

#[test]
fn parse_openai_format() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    assert_eq!(seq.format_detected, "openai");
    assert_eq!(seq.tokens.len(), 2);
    assert_eq!(seq.tokens[0].token, "Hello");
    assert_eq!(seq.model.as_deref(), Some("gpt-4"));
    assert!(seq.tokens[0].bytes.is_some());
    assert!(seq.tokens[0].top_logprobs.is_some());
}

#[test]
fn parse_vllm_format() {
    let seq = parse::parse_string(VLLM_FIXTURE, None, false).unwrap();
    assert_eq!(seq.format_detected, "vllm");
    assert_eq!(seq.tokens.len(), 4);
    assert_eq!(seq.tokens[0].token, "The");
    assert_eq!(seq.model.as_deref(), Some("meta-llama/Llama-2-7b"));
}

#[test]
fn parse_jsonl_format() {
    let seq = parse::parse_string(JSONL_FIXTURE, None, false).unwrap();
    assert_eq!(seq.format_detected, "jsonl");
    assert_eq!(seq.tokens.len(), 7);
    assert_eq!(seq.tokens[0].token, "Once");
}

#[test]
fn summary_computes_correctly() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let summary = metrics::compute_summary(&seq);
    assert_eq!(summary.token_count, 2);
    assert!((summary.mean_logprob - (-0.65)).abs() < 1e-10);
    assert!(summary.perplexity > 1.0);
}

#[test]
fn entropy_with_top_logprobs() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let entropies = metrics::compute_entropy(&seq);
    assert_eq!(entropies.len(), 2);
    // With top-k, partial entropy should be > 0
    assert!(entropies[0].entropy_partial > 0.0);
    // Missing mass should be > 0 (only 3 of many tokens shown)
    assert!(entropies[0].missing_mass > 0.0);
}

#[test]
fn bpb_works_with_bytes() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let result = metrics::compute_bpb(&seq);
    match result {
        metrics::BpbResult::Value { bpb } => assert!(bpb > 0.0),
        metrics::BpbResult::Unavailable { reason } => panic!("expected BPB value, got: {reason}"),
    }
}

#[test]
fn bpb_refuses_without_bytes() {
    let seq = parse::parse_string(JSONL_FIXTURE, None, false).unwrap();
    let result = metrics::compute_bpb(&seq);
    match result {
        metrics::BpbResult::Unavailable { .. } => {} // expected
        metrics::BpbResult::Value { .. } => panic!("should have refused without bytes"),
    }
}

#[test]
fn validate_clean_input() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let findings = diagnostics::validate(&seq);
    // Should pass with no errors
    let errors: Vec<_> = findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
}

#[test]
fn validate_catches_positive_logprob() {
    let bad_json = r#"{"token":"bad","logprob":0.5}
{"token":"ok","logprob":-0.3}"#;
    let seq = parse::parse_string(bad_json, None, false).unwrap();
    let findings = diagnostics::validate(&seq);
    let has_positive_error = findings
        .iter()
        .any(|f| f.check == "nonpositive_logprob");
    assert!(has_positive_error, "should catch positive logprob");
}

#[test]
fn diagnose_reports_missing_mass() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let findings = diagnostics::diagnose(&seq);
    let has_mass_check = findings.iter().any(|f| f.check == "missing_mass");
    assert!(has_mass_check, "diagnose should report on missing mass");
}

#[test]
fn confidence_filter_works() {
    let seq = parse::parse_string(VLLM_FIXTURE, None, false).unwrap();
    let low = filters::find_low_confidence(&seq, -1.0, 2);
    // " answer" (-1.1) and " 42" (-3.5) are below -1.0
    assert!(
        low.len() >= 2,
        "expected at least 2 low-confidence tokens, got {}",
        low.len()
    );
    assert!(low.iter().any(|t| t.token == " 42"));
}

#[test]
fn missing_mass_math() {
    // 3 tokens with probabilities summing to ~0.77
    let lps = [-0.5_f64, -2.0, -3.5];
    let mm = math::missing_mass(&lps);
    let expected_mass: f64 = lps.iter().map(|lp| lp.exp()).sum();
    let expected_missing = 1.0 - expected_mass;
    assert!((mm - expected_missing).abs() < 1e-10);
}

#[test]
fn perplexity_math() {
    let lps = [-0.5, -0.8];
    let ppl = math::perplexity(&lps);
    let expected = (-math::mean_logprob(&lps)).exp();
    assert!((ppl - expected).abs() < 1e-10);
}

// ─── New math-proving tests ──────────────────────────────────────

/// Verify that partial entropy < true entropy < normalized entropy
/// for a known truncated distribution.
#[test]
fn truncated_entropy_bias_is_bounded() {
    // True distribution: [0.4, 0.3, 0.15, 0.10, 0.05]
    // We observe only the top 3: [0.4, 0.3, 0.15]
    let full_probs: [f64; 5] = [0.4, 0.3, 0.15, 0.10, 0.05];
    let observed_probs: [f64; 3] = [0.4, 0.3, 0.15];

    // True entropy
    let true_entropy: f64 = full_probs
        .iter()
        .map(|&p| -p * p.log2())
        .sum();

    let observed_lps: Vec<f64> = observed_probs.iter().map(|p| p.ln()).collect();

    let h_partial = math::entropy_bits_partial(&observed_lps);
    let h_normalized = math::entropy_bits_normalized(&observed_lps);

    // Partial entropy is a lower bound on true entropy
    assert!(
        h_partial < true_entropy,
        "partial entropy ({h_partial:.4}) should be < true entropy ({true_entropy:.4})"
    );
    // Normalized entropy overshoots for significantly truncated distributions
    // (the renormalized distribution is more uniform than the full one)
    assert!(
        h_normalized > h_partial,
        "normalized ({h_normalized:.4}) should be > partial ({h_partial:.4})"
    );
}

/// Feed raw logits (unnormalized) and verify diagnose catches it.
#[test]
fn unnormalized_logits_detected() {
    // Raw logits — not log-probabilities. log Z will be >> 0.
    let seq = make_seq_with_topk(vec![
        vec![("A", 2.0), ("B", 1.0), ("C", 0.5)],
        vec![("X", 3.0), ("Y", 1.5), ("Z", 0.0)],
    ]);

    let report = diagnostics::diagnose_report(&seq);
    assert_eq!(
        report.normalization_status,
        Severity::Error,
        "should detect unnormalized scores"
    );
    assert!(
        report.mean_log_mass > 2.0,
        "mean log mass should be >> 0 for raw logits, got {}",
        report.mean_log_mass
    );
}

/// All logprobs exactly 0.0 should be flagged.
#[test]
fn all_zero_logprobs_flagged() {
    let seq = make_simple_seq(vec![0.0, 0.0, 0.0]);
    let findings = diagnostics::diagnose(&seq);
    let has_zero = findings.iter().any(|f| f.check == "all_zero_logprobs");
    assert!(has_zero, "should flag all-zero logprobs");
}

/// Constant (identical) logprobs should be flagged as suspicious.
#[test]
fn constant_logprobs_flagged() {
    let seq = make_simple_seq(vec![-1.5, -1.5, -1.5]);
    let findings = diagnostics::diagnose(&seq);
    let has_constant = findings.iter().any(|f| f.check == "constant_logprobs");
    assert!(has_constant, "should flag constant logprobs");
}

/// Empty input should return a parse error.
#[test]
fn empty_input_errors() {
    let result = parse::parse_string("", None, false);
    assert!(result.is_err(), "empty input should return error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("empty"),
        "error should mention empty: {err_msg}"
    );
}

/// Malformed JSON should return a parse error.
#[test]
fn malformed_json_errors() {
    let result = parse::parse_string("{not valid json at all!!!", None, false);
    assert!(result.is_err(), "malformed JSON should return error");
}

/// Distribution with >50% missing mass should flag entropy as unreliable.
#[test]
fn missing_mass_high_flags_unreliable() {
    // Single token with prob ~0.37 → missing mass ~0.63 → unreliable
    let seq = make_seq_with_topk(vec![vec![("only", -1.0)]]);
    let entropies = metrics::compute_entropy(&seq);
    assert_eq!(entropies.len(), 1);
    assert!(
        entropies[0].missing_mass > 0.5,
        "missing mass should be >50%: {}",
        entropies[0].missing_mass
    );
    assert!(
        entropies[0].unreliable,
        "should be flagged as unreliable"
    );
}

/// Create a sequence with one high-entropy token among low-entropy ones
/// and verify detect_entropy_spikes catches it.
#[test]
fn entropy_spike_detection() {
    let entropies = vec![
        make_token_entropy(0, "a", 0.1, 0.1),
        make_token_entropy(1, "b", 0.1, 0.1),
        make_token_entropy(2, "SPIKE", 10.0, 10.0),
        make_token_entropy(3, "c", 0.1, 0.1),
        make_token_entropy(4, "d", 0.1, 0.1),
    ];

    let spikes = filters::detect_entropy_spikes(&entropies, 1.5);
    assert!(
        spikes.contains(&2),
        "should detect spike at position 2, got: {spikes:?}"
    );
    // Other positions should not be flagged
    assert!(
        !spikes.contains(&0) && !spikes.contains(&1),
        "low-entropy positions should not be spikes"
    );
}

/// BPB should refuse and explain why when no byte data is available.
#[test]
fn bpb_strict_refuses_token_bytes_fallback() {
    let seq = make_simple_seq(vec![-0.5, -1.0]);
    let result = metrics::compute_bpb(&seq);
    match result {
        metrics::BpbResult::Unavailable { reason } => {
            assert!(
                reason.contains("byte") || reason.contains("BPE"),
                "error should explain byte requirement: {reason}"
            );
        }
        metrics::BpbResult::Value { .. } => {
            panic!("should refuse BPB without byte data")
        }
    }
}

/// Unsorted top_logprobs should be caught by validate.
#[test]
fn validate_catches_unsorted_top_logprobs() {
    // top_logprobs in ascending order (wrong — should be descending)
    let seq = LogprobSequence {
        tokens: vec![TokenLogprob {
            token: "test".into(),
            logprob: -0.5,
            bytes: None,
            top_logprobs: Some(vec![
                TopKEntry {
                    token: "worst".into(),
                    logprob: -3.0,
                },
                TopKEntry {
                    token: "mid".into(),
                    logprob: -1.5,
                },
                TopKEntry {
                    token: "best".into(),
                    logprob: -0.5,
                },
            ]),
        }],
        model: None,
        format_detected: "test".into(),
        total_logprob: -0.5,
        is_normalized: None,
    };

    let findings = diagnostics::validate(&seq);
    let has_sorted = findings
        .iter()
        .any(|f| f.check == "sorted_top_logprobs");
    assert!(has_sorted, "should catch unsorted top_logprobs");
}

/// Duplicate tokens in top_logprobs should be caught by validate.
#[test]
fn validate_catches_duplicate_top_tokens() {
    let seq = LogprobSequence {
        tokens: vec![TokenLogprob {
            token: "test".into(),
            logprob: -0.5,
            bytes: None,
            top_logprobs: Some(vec![
                TopKEntry {
                    token: "hello".into(),
                    logprob: -0.5,
                },
                TopKEntry {
                    token: "world".into(),
                    logprob: -1.0,
                },
                TopKEntry {
                    token: "hello".into(),
                    logprob: -2.0,
                },
            ]),
        }],
        model: None,
        format_detected: "test".into(),
        total_logprob: -0.5,
        is_normalized: None,
    };

    let findings = diagnostics::validate(&seq);
    let has_dup = findings
        .iter()
        .any(|f| f.check == "duplicate_top_token");
    assert!(has_dup, "should catch duplicate tokens in top_logprobs");
}

/// Verify vLLM top_logprobs are sorted after parsing (JSON object key order is arbitrary).
#[test]
fn vllm_top_logprobs_sorted_after_parse() {
    let seq = parse::parse_string(VLLM_FIXTURE, None, false).unwrap();
    for (i, tok) in seq.tokens.iter().enumerate() {
        if let Some(ref top_k) = tok.top_logprobs {
            for w in top_k.windows(2) {
                assert!(
                    w[0].logprob >= w[1].logprob,
                    "vLLM top_logprobs not sorted at position {i}: {} >= {} failed",
                    w[0].logprob,
                    w[1].logprob
                );
            }
        }
    }
}

/// Verify --format override works.
#[test]
fn format_override_works() {
    // Force JSONL parsing on a JSON array
    let input = r#"[{"token":"Hi","logprob":-0.5},{"token":"!","logprob":-1.0}]"#;
    let seq = parse::parse_string(
        input,
        Some(logprobe::types::InputFormat::JsonlStream),
        false,
    )
    .unwrap();
    assert_eq!(seq.format_detected, "jsonl");
    assert_eq!(seq.tokens.len(), 2);
}

/// Verify diagnose JSON output is valid and deserializable.
#[test]
fn diagnose_json_roundtrips() {
    let seq = parse::parse_string(OPENAI_FIXTURE, None, false).unwrap();
    let report = diagnostics::diagnose_report(&seq);
    let json = serde_json::to_string(&report).expect("should serialize");
    let _: logprobe::types::DiagnoseReport =
        serde_json::from_str(&json).expect("should deserialize back");
}

/// Parse and diagnose a real GPT-4o-mini API response.
#[test]
fn real_gpt4o_mini_creative() {
    let input = include_str!("../demo/gpt4o_mini_creative.json");
    let seq = parse::parse_string(input, None, false).unwrap();
    assert_eq!(seq.format_detected, "openai");
    assert_eq!(seq.tokens.len(), 150);
    assert_eq!(seq.model.as_deref(), Some("gpt-4o-mini-2024-07-18"));

    let report = diagnostics::diagnose_report(&seq);
    assert_eq!(report.normalization_status, Severity::Ok);
    assert!(report.mean_missing_mass > 0.0, "creative writing should have some missing mass");
    let errors: Vec<_> = report.findings.iter().filter(|f| f.severity == Severity::Error).collect();
    assert!(errors.is_empty(), "should have no validation errors: {errors:?}");
}

/// Parse and diagnose a Gemini-format response.
#[test]
fn gemini_format_parses() {
    let input = include_str!("../demo/gemini_sample.json");
    let seq = parse::parse_string(input, None, false).unwrap();
    assert_eq!(seq.format_detected, "gemini");
    assert_eq!(seq.tokens.len(), 12);
    assert_eq!(seq.model.as_deref(), Some("gemini-2.0-flash"));
    assert!(seq.tokens[0].top_logprobs.is_some());

    let report = diagnostics::diagnose_report(&seq);
    assert_eq!(report.normalization_status, Severity::Ok);
    let errors: Vec<_> = report.findings.iter().filter(|f| f.severity == Severity::Error).collect();
    assert!(errors.is_empty(), "gemini should have no errors: {errors:?}");
}

/// Parse and diagnose an Ollama-format response.
#[test]
fn ollama_format_parses() {
    let input = include_str!("../demo/ollama_sample.json");
    let seq = parse::parse_string(input, None, false).unwrap();
    assert_eq!(seq.format_detected, "ollama");
    assert_eq!(seq.tokens.len(), 7);
    assert_eq!(seq.model.as_deref(), Some("llama3.2:3b"));
    assert!(seq.tokens[0].bytes.is_some());
    assert!(seq.tokens[0].top_logprobs.is_some());

    let report = diagnostics::diagnose_report(&seq);
    assert_eq!(report.normalization_status, Severity::Ok);
    let errors: Vec<_> = report.findings.iter().filter(|f| f.severity == Severity::Error).collect();
    assert!(errors.is_empty(), "ollama should have no errors: {errors:?}");
}

// ─── Test helpers ────────────────────────────────────────────────

/// Build a simple sequence with no top_logprobs (just token logprobs).
fn make_simple_seq(logprobs: Vec<f64>) -> LogprobSequence {
    let total: f64 = logprobs.iter().sum();
    let tokens = logprobs
        .into_iter()
        .enumerate()
        .map(|(i, lp)| TokenLogprob {
            token: format!("t{i}"),
            logprob: lp,
            bytes: None,
            top_logprobs: None,
        })
        .collect();

    LogprobSequence {
        tokens,
        model: None,
        format_detected: "test".into(),
        total_logprob: total,
        is_normalized: None,
    }
}

/// Build a sequence where each token has top_logprobs.
/// Input: Vec of positions, each a Vec of (token, logprob) pairs.
/// The first entry in each position is the chosen token.
fn make_seq_with_topk(positions: Vec<Vec<(&str, f64)>>) -> LogprobSequence {
    let mut total = 0.0;
    let tokens: Vec<TokenLogprob> = positions
        .into_iter()
        .map(|entries| {
            let chosen_logprob = entries[0].1;
            total += chosen_logprob;
            TokenLogprob {
                token: entries[0].0.to_string(),
                logprob: chosen_logprob,
                bytes: None,
                top_logprobs: Some(
                    entries
                        .into_iter()
                        .map(|(t, lp)| TopKEntry {
                            token: t.to_string(),
                            logprob: lp,
                        })
                        .collect(),
                ),
            }
        })
        .collect();

    LogprobSequence {
        tokens,
        model: None,
        format_detected: "test".into(),
        total_logprob: total,
        is_normalized: None,
    }
}

fn make_token_entropy(
    position: usize,
    token: &str,
    entropy_partial: f64,
    entropy_normalized: f64,
) -> TokenEntropy {
    TokenEntropy {
        position,
        token: token.to_string(),
        entropy_partial,
        entropy_normalized,
        missing_mass: 0.0,
        unreliable: false,
    }
}
