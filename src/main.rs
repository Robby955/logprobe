use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

use logprobe::cli::{Cli, Command};
use logprobe::diagnostics;
use logprobe::filters;
use logprobe::metrics;
use logprobe::output;
use logprobe::parse;
use logprobe::types::{BatchResult, InputFormat, Severity};

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Batch has its own input handling — process and return early.
    if let Command::Batch { ref path } = cli.command {
        let results = run_batch(path, &cli)?;
        output::print_batch(&results, cli.json)?;
        return Ok(());
    }

    let input_data = match &cli.input {
        Some(path) => fs::read_to_string(path)
            .with_context(|| format!("failed to read input file: {path}"))?,
        None => {
            let mut buf = String::new();
            io::stdin()
                .read_to_string(&mut buf)
                .context("failed to read from stdin")?;
            buf
        }
    };

    let format_override = cli
        .format
        .as_ref()
        .map(|f| f.parse::<InputFormat>())
        .transpose()
        .map_err(|e| anyhow::anyhow!(e))?;

    let seq = parse::parse_string(&input_data, format_override, cli.strict_format)?;

    match cli.command {
        Command::Summary => {
            let summary = metrics::compute_summary(&seq);
            output::print_summary(&summary, cli.json)?;
        }
        Command::Entropy => {
            let entropies = metrics::compute_entropy(&seq);
            output::print_entropy(&entropies, cli.json)?;
        }
        Command::Confidence {
            threshold,
            context,
        } => {
            let low = filters::find_low_confidence(&seq, threshold, context);
            output::print_confidence(&low, threshold, cli.json)?;
        }
        Command::Bpb => {
            let result = metrics::compute_bpb(&seq);
            output::print_bpb(&result, cli.json)?;
        }
        Command::Highlight => {
            output::print_highlight(&seq)?;
        }
        Command::Validate => {
            let findings = diagnostics::validate(&seq);
            output::print_diagnostics(&findings, "validate", cli.json)?;
        }
        Command::Diagnose => {
            let report = diagnostics::diagnose_report(&seq);
            output::print_diagnose_report(&report, cli.json)?;
        }
        Command::Compare { ref other } => {
            let other_data = fs::read_to_string(other)
                .with_context(|| format!("failed to read comparison file: {other}"))?;
            let seq_b = parse::parse_string(&other_data, format_override, cli.strict_format)?;

            // clap assigns the first positional after the subcommand to `other`,
            // and the second to the global `input`. `seq` comes from `input` (the
            // second file the user typed), so it becomes file B in the comparison.
            let label_other = other.as_str();
            let label_input = cli.input.as_deref().unwrap_or("stdin");
            let report =
                metrics::compute_compare(&seq_b, &seq, label_other, label_input);
            output::print_compare(&report, cli.json)?;
        }
        Command::Batch { .. } => unreachable!("handled above"),
    }

    Ok(())
}

/// Collect .json file paths from a directory or treat the path as a single file.
fn collect_json_files(path: &str) -> Result<Vec<String>> {
    let p = Path::new(path);
    if p.is_dir() {
        let mut files: Vec<String> = fs::read_dir(p)
            .with_context(|| format!("failed to read directory: {path}"))?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let file_path = entry.path();
                if file_path.extension().and_then(|e| e.to_str()) == Some("json") {
                    Some(file_path.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        files.sort();
        if files.is_empty() {
            anyhow::bail!("no .json files found in directory: {path}");
        }
        Ok(files)
    } else if p.is_file() {
        Ok(vec![path.to_string()])
    } else {
        anyhow::bail!("path does not exist: {path}");
    }
}

/// Process a single file into a BatchResult, capturing errors gracefully.
fn process_one_file(file_path: &str, cli: &Cli) -> BatchResult {
    let file_name = Path::new(file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| file_path.to_string());

    let error_result = |err: String| BatchResult {
        file: file_name.clone(),
        model: None,
        format: String::new(),
        tokens: 0,
        perplexity: 0.0,
        mean_logprob: 0.0,
        missing_mass: None,
        entropy_bias: None,
        normalization: String::new(),
        bpb: None,
        error: Some(err),
    };

    let input_data = match fs::read_to_string(file_path) {
        Ok(data) => data,
        Err(e) => return error_result(format!("read error: {e}")),
    };

    let format_override = cli
        .format
        .as_ref()
        .map(|f| f.parse::<InputFormat>())
        .transpose();
    let format_override = match format_override {
        Ok(f) => f,
        Err(e) => return error_result(format!("format error: {e}")),
    };

    let seq = match parse::parse_string(&input_data, format_override, cli.strict_format) {
        Ok(s) => s,
        Err(e) => return error_result(format!("parse error: {e}")),
    };

    let summary = metrics::compute_summary(&seq);
    let report = diagnostics::diagnose_report(&seq);

    let normalization = match report.normalization_status {
        Severity::Ok => "pass",
        Severity::Warning => "warn",
        Severity::Error => "fail",
    };

    let missing_mass = if report.total_positions > 0 {
        Some(report.mean_missing_mass)
    } else {
        None
    };

    let entropy_bias = if report.total_positions > 0 {
        Some(report.entropy_bias)
    } else {
        None
    };

    let bpb = match metrics::compute_bpb(&seq) {
        metrics::BpbResult::Value { bpb } => Some(bpb),
        metrics::BpbResult::Unavailable { .. } => None,
    };

    BatchResult {
        file: file_name,
        model: seq.model.clone(),
        format: seq.format_detected.clone(),
        tokens: summary.token_count,
        perplexity: summary.perplexity,
        mean_logprob: summary.mean_logprob,
        missing_mass,
        entropy_bias,
        normalization: normalization.to_string(),
        bpb,
        error: None,
    }
}

/// Run batch processing over all files matching the path.
fn run_batch(path: &str, cli: &Cli) -> Result<Vec<BatchResult>> {
    let files = collect_json_files(path)?;
    let results: Vec<BatchResult> = files
        .iter()
        .map(|f| process_one_file(f, cli))
        .collect();
    Ok(results)
}
