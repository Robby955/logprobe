//! `logprobe` — detect normalization errors, entropy bias, and truncation
//! artifacts in LLM logprob data.
//!
//! This crate backs the `logprobe` command-line tool. It parses logprob output
//! from several providers (OpenAI, vLLM, Gemini, Ollama, and JSONL streams),
//! then computes diagnostics over the parsed [`types::LogprobSequence`]:
//! perplexity and summary statistics, per-token entropy, missing-mass and
//! normalization checks, bits-per-byte, and low-confidence highlighting.
//!
//! The modules mirror the CLI subcommands: [`parse`] reads input, [`metrics`]
//! and [`math`] compute statistics, [`diagnostics`] runs the `diagnose`/
//! `validate` checks, [`filters`] backs `confidence`/`highlight`, and
//! [`output`] renders human-readable and JSON results.
//!
//! # Example
//!
//! ```
//! use logprobe::{diagnostics, metrics, parse};
//!
//! // A tiny two-token sequence in the JSONL-array form.
//! let json = r#"[{"token":"Hi","logprob":-0.5},{"token":"!","logprob":-1.0}]"#;
//! let seq = parse::parse_string(json, None, false).unwrap();
//!
//! let summary = metrics::compute_summary(&seq);
//! assert_eq!(summary.token_count, 2);
//!
//! // Without top_logprobs, normalization cannot be assessed.
//! let report = diagnostics::diagnose_report(&seq);
//! assert_eq!(report.total_positions, 0);
//! ```

pub mod cli;
pub mod diagnostics;
pub mod filters;
pub mod math;
pub mod metrics;
pub mod output;
pub mod parse;
pub mod types;
