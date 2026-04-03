use serde::{Deserialize, Serialize};

/// A single token with its log-probability and optional top-k alternatives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopKEntry>>,
}

/// One entry in a top-k logprob distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopKEntry {
    pub token: String,
    pub logprob: f64,
}

/// A parsed sequence of token logprobs with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobSequence {
    pub tokens: Vec<TokenLogprob>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub format_detected: String,
    pub total_logprob: f64,
}

/// Supported input formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    OpenAI,
    VllmFlat,
    JsonlStream,
    Gemini,
    Ollama,
}

impl std::fmt::Display for InputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputFormat::OpenAI => write!(f, "openai"),
            InputFormat::VllmFlat => write!(f, "vllm"),
            InputFormat::JsonlStream => write!(f, "jsonl"),
            InputFormat::Gemini => write!(f, "gemini"),
            InputFormat::Ollama => write!(f, "ollama"),
        }
    }
}

impl std::str::FromStr for InputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" | "azure" | "deepseek" | "mistral" | "fireworks" | "grok" | "xai" => {
                Ok(InputFormat::OpenAI)
            }
            "vllm" | "together" => Ok(InputFormat::VllmFlat),
            "jsonl" | "stream" => Ok(InputFormat::JsonlStream),
            "gemini" | "google" | "vertex" => Ok(InputFormat::Gemini),
            "ollama" => Ok(InputFormat::Ollama),
            _ => Err(format!(
                "unknown format: {s} (expected: openai, vllm, jsonl, gemini, ollama)"
            )),
        }
    }
}

/// Structured report from the `diagnose` command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnoseReport {
    pub normalization_status: Severity,
    pub mean_log_mass: f64,
    pub max_log_mass: f64,
    pub mean_missing_mass: f64,
    pub high_missing_mass_count: usize,
    pub total_positions: usize,
    pub entropy_partial: f64,
    pub entropy_normalized: f64,
    pub entropy_bias: f64,
    pub has_bytes: bool,
    pub findings: Vec<DiagnosticFinding>,
    pub suspicious_patterns: Vec<DiagnosticFinding>,
}

/// Severity level for diagnostic findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Ok,
    Warning,
    Error,
}

/// A single diagnostic finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticFinding {
    pub severity: Severity,
    pub check: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<usize>,
}

/// Per-token entropy metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEntropy {
    pub position: usize,
    pub token: String,
    pub entropy_partial: f64,
    pub entropy_normalized: f64,
    pub missing_mass: f64,
    pub unreliable: bool,
}

/// Summary statistics for a sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceSummary {
    pub token_count: usize,
    pub mean_logprob: f64,
    pub total_logprob: f64,
    pub perplexity: f64,
    pub assumed_normalized: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_missing_mass: Option<f64>,
}

/// A low-confidence token with context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LowConfidenceToken {
    pub position: usize,
    pub token: String,
    pub logprob: f64,
    pub probability: f64,
    pub context_before: Vec<String>,
    pub context_after: Vec<String>,
}
