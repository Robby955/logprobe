use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{self, Read};

use logprobe::cli::{Cli, Command};
use logprobe::diagnostics;
use logprobe::filters;
use logprobe::metrics;
use logprobe::output;
use logprobe::parse;
use logprobe::types::InputFormat;

fn main() -> Result<()> {
    let cli = Cli::parse();

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
    }

    Ok(())
}
