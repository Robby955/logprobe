use anyhow::{bail, Result};
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use crossterm::ExecutableCommand;
use std::io::{self, Write};

use crate::metrics::BpbResult;
use crate::types::{
    DiagnoseReport, DiagnosticFinding, LogprobSequence, LowConfidenceToken, SequenceSummary,
    Severity, TokenEntropy,
};

pub fn print_summary(summary: &SequenceSummary, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(summary)?);
        return Ok(());
    }

    println!("=== Summary ===");
    println!("Tokens:           {}", summary.token_count);
    println!("Mean logprob:     {:.6}", summary.mean_logprob);
    println!("Total logprob:    {:.6}", summary.total_logprob);
    println!("Perplexity:       {:.4}", summary.perplexity);
    println!(
        "Normalized:       {}",
        if summary.assumed_normalized {
            "yes"
        } else {
            "unknown — run diagnose to check"
        }
    );
    if let Some(mm) = summary.mean_missing_mass {
        println!("Mean missing mass: {:.4}", mm);
    }
    Ok(())
}

pub fn print_entropy(entropies: &[TokenEntropy], json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(entropies)?);
        return Ok(());
    }

    println!(
        "{:<6} {:<20} {:>10} {:>10} {:>10} Flag",
        "Pos", "Token", "H_partial", "H_norm", "Missing"
    );
    println!("{}", "-".repeat(72));

    for e in entropies {
        let flag = if e.unreliable { "UNRELIABLE" } else { "" };
        let token_display = truncate_token(&e.token, 18);
        println!(
            "{:<6} {:<20} {:>10.4} {:>10.4} {:>10.4} {}",
            e.position, token_display, e.entropy_partial, e.entropy_normalized, e.missing_mass,
            flag
        );
    }

    Ok(())
}

pub fn print_confidence(tokens: &[LowConfidenceToken], threshold: f64, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(tokens)?);
        return Ok(());
    }

    if tokens.is_empty() {
        println!("No tokens below threshold {threshold}");
        return Ok(());
    }

    println!(
        "=== Low-confidence tokens (threshold: {threshold}) ==="
    );
    println!();

    for tok in tokens {
        let ctx_before = tok.context_before.join("");
        let ctx_after = tok.context_after.join("");
        println!(
            "Position {}: logprob={:.4} (p={:.6})",
            tok.position, tok.logprob, tok.probability
        );
        println!(
            "  Context: ...{}[{}]{}...",
            ctx_before, tok.token, ctx_after
        );
        println!();
    }

    println!("Total: {} tokens below threshold", tokens.len());
    Ok(())
}

pub fn print_bpb(result: &BpbResult, json: bool) -> Result<()> {
    match result {
        BpbResult::Value { bpb } => {
            if json {
                println!("{}", serde_json::to_string_pretty(result)?);
            } else {
                println!("Bits per byte: {bpb:.6}");
            }
        }
        BpbResult::Unavailable { reason } => {
            if json {
                println!("{}", serde_json::to_string_pretty(result)?);
            } else {
                bail!("{reason}");
            }
        }
    }
    Ok(())
}

pub fn print_highlight(seq: &LogprobSequence) -> Result<()> {
    let no_color = std::env::var("NO_COLOR").is_ok();
    let mut stdout = io::stdout();

    if no_color {
        // Plain text fallback: [logprob] token
        for tok in &seq.tokens {
            print!("{}", tok.token);
        }
        println!();
    } else {
        for tok in &seq.tokens {
            let color = logprob_to_color(tok.logprob);
            stdout.execute(SetForegroundColor(color))?;
            stdout.execute(Print(&tok.token))?;
        }
        stdout.execute(ResetColor)?;
        println!();
        println!();
        print_legend(&mut stdout)?;
    }

    Ok(())
}

pub fn print_diagnostics(
    findings: &[DiagnosticFinding],
    command: &str,
    json: bool,
) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(findings)?);
        return Ok(());
    }

    println!("=== {command} ===");
    println!();

    let errors = findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .count();
    let warnings = findings
        .iter()
        .filter(|f| f.severity == Severity::Warning)
        .count();

    for finding in findings {
        let prefix = match finding.severity {
            Severity::Ok => "[OK]",
            Severity::Warning => "[WARN]",
            Severity::Error => "[ERROR]",
        };
        let pos = finding
            .position
            .map(|p| format!(" (position {p})"))
            .unwrap_or_default();
        println!("{prefix} {}{pos}: {}", finding.check, finding.message);
    }

    println!();
    if errors > 0 {
        println!("Result: {errors} error(s), {warnings} warning(s)");
    } else if warnings > 0 {
        println!("Result: {warnings} warning(s), no errors");
    } else {
        println!("Result: all checks passed");
    }

    Ok(())
}

pub fn print_diagnose_report(report: &DiagnoseReport, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    // Normalization line
    if report.total_positions > 0 {
        match report.normalization_status {
            Severity::Error => {
                println!(
                    "Normalization:  FAIL (log mass = {:.4} — likely raw logits)",
                    report.mean_log_mass
                );
            }
            Severity::Warning => {
                println!(
                    "Normalization:  WARN (log mass = {:.4} — possible normalization issue)",
                    report.mean_log_mass
                );
            }
            Severity::Ok => {
                println!(
                    "Normalization:  pass (log mass = {:.4})",
                    report.mean_log_mass
                );
            }
        }
    } else {
        println!("Normalization:  unknown (no top_logprobs data)");
    }

    // Missing mass line
    if report.total_positions > 0 {
        if report.high_missing_mass_count > 0 {
            println!(
                "Missing mass:   {:.4} ({}/{} positions >50% missing)",
                report.mean_missing_mass,
                report.high_missing_mass_count,
                report.total_positions
            );
        } else {
            println!(
                "Missing mass:   {:.4} ({} positions)",
                report.mean_missing_mass, report.total_positions
            );
        }
    }

    // Entropy bias line — signed value is informative
    if report.total_positions > 0 {
        println!(
            "Entropy bias:   {:+.4} bits (partial: {:.4}, normalized: {:.4})",
            report.entropy_bias, report.entropy_partial, report.entropy_normalized
        );
    }

    // BPB reliability
    if report.has_bytes {
        println!("BPB:            byte data available");
    } else {
        println!("BPB:            no byte data (cannot compute)");
    }

    // Validation summary
    let error_count = report
        .findings
        .iter()
        .filter(|f| f.severity == Severity::Error)
        .count();
    let token_count_msg = report
        .findings
        .iter()
        .find(|f| f.check == "all_checks")
        .map(|f| f.message.clone());

    println!();
    if error_count > 0 {
        println!(
            "Validation: {} error(s) found",
            error_count
        );
        for f in &report.findings {
            if f.severity == Severity::Error {
                let pos = f
                    .position
                    .map(|p| format!(" (position {p})"))
                    .unwrap_or_default();
                println!("  [ERROR] {}{pos}: {}", f.check, f.message);
            }
        }
    } else if let Some(msg) = token_count_msg {
        println!("Validation: {msg}");
    }

    // Suspicious patterns
    if !report.suspicious_patterns.is_empty() {
        println!();
        for pattern in &report.suspicious_patterns {
            let prefix = match pattern.severity {
                Severity::Ok => "[OK]",
                Severity::Warning => "[WARN]",
                Severity::Error => "[ERROR]",
            };
            println!("{prefix} {}: {}", pattern.check, pattern.message);
        }
    }

    Ok(())
}

/// Map a logprob to a terminal color: green (confident) -> yellow -> red (uncertain).
fn logprob_to_color(logprob: f64) -> Color {
    if logprob > -0.1 {
        Color::Green
    } else if logprob > -0.5 {
        Color::DarkGreen
    } else if logprob > -1.0 {
        Color::Yellow
    } else if logprob > -2.0 {
        Color::DarkYellow
    } else if logprob > -5.0 {
        Color::Red
    } else {
        Color::DarkRed
    }
}

fn print_legend(stdout: &mut impl Write) -> Result<()> {
    let labels = [
        (Color::Green, "> -0.1"),
        (Color::DarkGreen, "-0.1 to -0.5"),
        (Color::Yellow, "-0.5 to -1.0"),
        (Color::DarkYellow, "-1.0 to -2.0"),
        (Color::Red, "-2.0 to -5.0"),
        (Color::DarkRed, "< -5.0"),
    ];
    print!("Legend: ");
    for (color, label) in &labels {
        stdout.execute(SetForegroundColor(*color))?;
        stdout.execute(Print(format!("{label}  ")))?;
    }
    stdout.execute(ResetColor)?;
    println!();
    Ok(())
}

fn truncate_token(token: &str, max_len: usize) -> String {
    let display: String = token
        .chars()
        .map(|c| match c {
            '\n' => '↵',
            '\r' => '⏎',
            '\t' => '→',
            c if c.is_control() => '·',
            c => c,
        })
        .collect();
    let char_count = display.chars().count();
    if char_count > max_len {
        let truncated: String = display.chars().take(max_len - 3).collect();
        format!("{truncated}...")
    } else {
        display
    }
}
