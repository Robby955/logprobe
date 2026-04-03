use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "logprobe",
    version,
    about = "LLM probability diagnostics CLI",
    long_about = "logprobe detects normalization errors, entropy bias, and truncation artifacts \
                  in LLM logprob data. Most tools treat top-k logprobs as a complete distribution — \
                  they aren't. logprobe quantifies exactly how wrong that assumption is.",
    after_help = "Examples:\n  \
                  logprobe diagnose response.json\n  \
                  logprobe validate response.json --json\n  \
                  logprobe entropy response.json\n  \
                  cat response.json | logprobe summary"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Input file (reads from stdin if omitted)
    #[arg(global = true)]
    pub input: Option<String>,

    /// Override auto-detected format (openai, vllm, jsonl)
    #[arg(long, global = true)]
    pub format: Option<String>,

    /// Error on ambiguous format detection instead of guessing
    #[arg(long, global = true)]
    pub strict_format: bool,

    /// Output as JSON
    #[arg(long, global = true)]
    pub json: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Detect normalization errors, entropy bias, and invalid distributions
    Diagnose,

    /// Check logprob data integrity (finite, sorted, no duplicates, mass <= 1)
    Validate,

    /// Sequence statistics (mean logprob, perplexity, missing mass)
    Summary,

    /// Per-token entropy from top_logprobs (partial and normalized)
    Entropy,

    /// Find low-confidence tokens with surrounding context
    Confidence {
        /// Log-probability threshold (tokens below this are flagged)
        #[arg(short, long, default_value = "-2.0", allow_hyphen_values = true)]
        threshold: f64,

        /// Number of context tokens to show before/after
        #[arg(short = 'C', long, default_value = "3")]
        context: usize,
    },

    /// Bits-per-byte (strict: requires explicit byte counts)
    Bpb,

    /// Terminal visualization colored by confidence
    Highlight,
}
