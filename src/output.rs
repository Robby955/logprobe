//! Rendering of command results to a writer.
//!
//! Every function is generic over its output sink (`W: Write`) and returns
//! [`io::Result`], so callers can target stdout, a buffer, or a file, and the
//! renderers can be unit-tested without capturing process stdout. No
//! application error type leaks out of this module.

use crossterm::ExecutableCommand;
use crossterm::style::{Color, Print, ResetColor, SetForegroundColor};
use serde::Serialize;
use std::io::{self, Write};

use crate::metrics::BpbResult;
use crate::types::{
    BatchResult, CompareReport, DiagnoseReport, DiagnosticFinding, LogprobSequence,
    LowConfidenceToken, SequenceSummary, Severity, TokenEntropy,
};

/// Serialize `value` to pretty JSON, mapping any serializer failure to an
/// [`io::Error`] so callers only deal with one error type.
fn json_pretty<T: Serialize>(value: &T) -> io::Result<String> {
    serde_json::to_string_pretty(value).map_err(io::Error::other)
}

/// Quote and escape a CSV field if it contains a comma, quote, or newline.
fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Render the sequence summary (mean logprob, perplexity, missing mass).
pub fn print_summary(w: &mut impl Write, summary: &SequenceSummary, json: bool) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(summary)?)?;
        return Ok(());
    }

    writeln!(w, "=== Summary ===")?;
    writeln!(w, "Tokens:           {}", summary.token_count)?;
    writeln!(w, "Mean logprob:     {:.6}", summary.mean_logprob)?;
    writeln!(w, "Total logprob:    {:.6}", summary.total_logprob)?;
    writeln!(w, "Perplexity:       {:.4}", summary.perplexity)?;
    writeln!(
        w,
        "Normalized:       {}",
        if summary.assumed_normalized {
            "yes"
        } else {
            "unknown — run diagnose to check"
        }
    )?;
    if let Some(mm) = summary.mean_missing_mass {
        writeln!(w, "Mean missing mass: {mm:.4}")?;
    }
    Ok(())
}

/// Render the per-token entropy table.
pub fn print_entropy(w: &mut impl Write, entropies: &[TokenEntropy], json: bool) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(&entropies)?)?;
        return Ok(());
    }

    writeln!(
        w,
        "{:<6} {:<20} {:>10} {:>10} {:>10} Flag",
        "Pos", "Token", "H_partial", "H_norm", "Missing"
    )?;
    writeln!(w, "{}", "-".repeat(72))?;

    for e in entropies {
        let flag = if e.unreliable { "UNRELIABLE" } else { "" };
        let token_display = truncate_token(&e.token, 18);
        writeln!(
            w,
            "{:<6} {:<20} {:>10.4} {:>10.4} {:>10.4} {}",
            e.position,
            token_display,
            e.entropy_partial,
            e.entropy_normalized,
            e.missing_mass,
            flag
        )?;
    }

    Ok(())
}

/// Render low-confidence tokens with surrounding context.
pub fn print_confidence(
    w: &mut impl Write,
    tokens: &[LowConfidenceToken],
    threshold: f64,
    json: bool,
) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(&tokens)?)?;
        return Ok(());
    }

    if tokens.is_empty() {
        writeln!(w, "No tokens below threshold {threshold}")?;
        return Ok(());
    }

    writeln!(w, "=== Low-confidence tokens (threshold: {threshold}) ===")?;
    writeln!(w)?;

    for tok in tokens {
        let ctx_before = tok.context_before.join("");
        let ctx_after = tok.context_after.join("");
        writeln!(
            w,
            "Position {}: logprob={:.4} (p={:.6})",
            tok.position, tok.logprob, tok.probability
        )?;
        writeln!(
            w,
            "  Context: ...{}[{}]{}...",
            ctx_before, tok.token, ctx_after
        )?;
        writeln!(w)?;
    }

    writeln!(w, "Total: {} tokens below threshold", tokens.len())?;
    Ok(())
}

/// Render the bits-per-byte result, or the reason it is unavailable.
///
/// The CLI exits non-zero when BPB is unavailable in non-JSON mode; that
/// decision is made by the caller, so this renderer only writes output.
pub fn print_bpb(w: &mut impl Write, result: &BpbResult, json: bool) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(result)?)?;
        return Ok(());
    }
    match result {
        BpbResult::Value { bpb } => writeln!(w, "Bits per byte: {bpb:.6}")?,
        BpbResult::Unavailable { reason } => writeln!(w, "{reason}")?,
    }
    Ok(())
}

/// Render the sequence with each token colored by its log-probability.
///
/// Honors the `NO_COLOR` environment variable by falling back to plain text.
pub fn print_highlight(w: &mut impl Write, seq: &LogprobSequence) -> io::Result<()> {
    let no_color = std::env::var("NO_COLOR").is_ok();

    if no_color {
        // Plain text fallback: just the concatenated tokens.
        for tok in &seq.tokens {
            write!(w, "{}", tok.token)?;
        }
        writeln!(w)?;
    } else {
        for tok in &seq.tokens {
            let color = logprob_to_color(tok.logprob);
            w.execute(SetForegroundColor(color))?;
            w.execute(Print(&tok.token))?;
        }
        w.execute(ResetColor)?;
        writeln!(w)?;
        writeln!(w)?;
        print_legend(w)?;
    }

    Ok(())
}

/// Render a flat list of diagnostic findings (`validate` output).
pub fn print_diagnostics(
    w: &mut impl Write,
    findings: &[DiagnosticFinding],
    command: &str,
    json: bool,
) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(&findings)?)?;
        return Ok(());
    }

    writeln!(w, "=== {command} ===")?;
    writeln!(w)?;

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
        writeln!(w, "{prefix} {}{pos}: {}", finding.check, finding.message)?;
    }

    writeln!(w)?;
    if errors > 0 {
        writeln!(w, "Result: {errors} error(s), {warnings} warning(s)")?;
    } else if warnings > 0 {
        writeln!(w, "Result: {warnings} warning(s), no errors")?;
    } else {
        writeln!(w, "Result: all checks passed")?;
    }

    Ok(())
}

/// Render the structured `diagnose` report.
pub fn print_diagnose_report(
    w: &mut impl Write,
    report: &DiagnoseReport,
    json: bool,
) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(report)?)?;
        return Ok(());
    }

    // Normalization line
    if report.total_positions > 0 {
        match report.normalization_status {
            Severity::Error => {
                writeln!(
                    w,
                    "Normalization:  FAIL (log mass = {:.4} — likely raw logits)",
                    report.mean_log_mass
                )?;
            }
            Severity::Warning => {
                writeln!(
                    w,
                    "Normalization:  WARN (log mass = {:.4} — possible normalization issue)",
                    report.mean_log_mass
                )?;
            }
            Severity::Ok => {
                writeln!(
                    w,
                    "Normalization:  pass (log mass = {:.4})",
                    report.mean_log_mass
                )?;
            }
        }
    } else {
        writeln!(w, "Normalization:  unknown (no top_logprobs data)")?;
    }

    // Missing mass line
    if report.total_positions > 0 {
        if report.high_missing_mass_count > 0 {
            writeln!(
                w,
                "Missing mass:   {:.4} ({}/{} positions >50% missing)",
                report.mean_missing_mass, report.high_missing_mass_count, report.total_positions
            )?;
        } else {
            writeln!(
                w,
                "Missing mass:   {:.4} ({} positions)",
                report.mean_missing_mass, report.total_positions
            )?;
        }
    }

    // Entropy bias line — signed value is informative
    if report.total_positions > 0 {
        writeln!(
            w,
            "Entropy bias:   {:+.4} bits (partial: {:.4}, normalized: {:.4})",
            report.entropy_bias, report.entropy_partial, report.entropy_normalized
        )?;
    }

    // BPB reliability
    if report.has_bytes {
        writeln!(w, "BPB:            byte data available")?;
    } else {
        writeln!(w, "BPB:            no byte data (cannot compute)")?;
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

    writeln!(w)?;
    if error_count > 0 {
        writeln!(w, "Validation: {error_count} error(s) found")?;
        for f in &report.findings {
            if f.severity == Severity::Error {
                let pos = f
                    .position
                    .map(|p| format!(" (position {p})"))
                    .unwrap_or_default();
                writeln!(w, "  [ERROR] {}{pos}: {}", f.check, f.message)?;
            }
        }
    } else if let Some(msg) = token_count_msg {
        writeln!(w, "Validation: {msg}")?;
    }

    // Suspicious patterns
    if !report.suspicious_patterns.is_empty() {
        writeln!(w)?;
        for pattern in &report.suspicious_patterns {
            let prefix = match pattern.severity {
                Severity::Ok => "[OK]",
                Severity::Warning => "[WARN]",
                Severity::Error => "[ERROR]",
            };
            writeln!(w, "{prefix} {}: {}", pattern.check, pattern.message)?;
        }
    }

    Ok(())
}

/// Render a side-by-side comparison of two sequences.
pub fn print_compare(w: &mut impl Write, report: &CompareReport, json: bool) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(report)?)?;
        return Ok(());
    }

    let no_color = std::env::var("NO_COLOR").is_ok();

    // Header
    let label_a = &report.file_a.label;
    let label_b = &report.file_b.label;
    let col_w = label_a.len().max(label_b.len()).max(14);

    writeln!(w, "=== Compare ===")?;
    writeln!(w)?;
    writeln!(
        w,
        "{:<20} {:>col_w$}   {:>col_w$}   {:>12}",
        "Metric", label_a, label_b, "Delta"
    )?;
    let rule_len = 20 + 3 + col_w + 3 + col_w + 3 + 12;
    writeln!(w, "{}", "\u{2500}".repeat(rule_len))?;

    // Tokens (no delta — counts are independent)
    writeln!(
        w,
        "{:<20} {:>col_w$}   {:>col_w$}",
        "Tokens", report.file_a.token_count, report.file_b.token_count,
    )?;

    // Model
    let model_a = report.file_a.model.as_deref().unwrap_or("-");
    let model_b = report.file_b.model.as_deref().unwrap_or("-");
    writeln!(
        w,
        "{:<20} {:>col_w$}   {:>col_w$}",
        "Model", model_a, model_b,
    )?;

    // Perplexity (lower is better)
    print_compare_row(
        w,
        "Perplexity",
        &format!("{:.4}", report.file_a.perplexity),
        &format!("{:.4}", report.file_b.perplexity),
        report.delta_perplexity,
        4,
        true,
        col_w,
        no_color,
    )?;

    // Mean logprob (higher/closer to 0 is better)
    print_compare_row(
        w,
        "Mean logprob",
        &format!("{:.6}", report.file_a.mean_logprob),
        &format!("{:.6}", report.file_b.mean_logprob),
        report.delta_mean_logprob,
        6,
        false,
        col_w,
        no_color,
    )?;

    // Mean entropy (lower is better)
    print_compare_row(
        w,
        "Mean entropy",
        &format!("{:.4}", report.file_a.mean_entropy_partial),
        &format!("{:.4}", report.file_b.mean_entropy_partial),
        report.delta_entropy_partial,
        4,
        true,
        col_w,
        no_color,
    )?;

    // Missing mass (lower is better, show as percentage)
    if let (Some(mm_a), Some(mm_b), Some(delta)) = (
        report.file_a.mean_missing_mass,
        report.file_b.mean_missing_mass,
        report.delta_missing_mass,
    ) {
        print_compare_row(
            w,
            "Missing mass",
            &format!("{:.2}%", mm_a * 100.0),
            &format!("{:.2}%", mm_b * 100.0),
            delta * 100.0,
            2,
            true,
            col_w,
            no_color,
        )?;
    }

    // BPB (lower is better)
    if let (Some(bpb_a), Some(bpb_b), Some(delta)) =
        (report.file_a.bpb, report.file_b.bpb, report.delta_bpb)
    {
        print_compare_row(
            w,
            "BPB",
            &format!("{bpb_a:.6}"),
            &format!("{bpb_b:.6}"),
            delta,
            6,
            true,
            col_w,
            no_color,
        )?;
    }

    writeln!(w)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn print_compare_row(
    w: &mut impl Write,
    label: &str,
    val_a: &str,
    val_b: &str,
    delta: f64,
    precision: usize,
    lower_is_better: bool,
    col_w: usize,
    no_color: bool,
) -> io::Result<()> {
    let sign = if delta > 0.0 { "+" } else { "" };
    let delta_str = format!("{sign}{delta:.precision$}");

    let improved = if lower_is_better {
        delta < -1e-12
    } else {
        delta > 1e-12
    };
    let worsened = if lower_is_better {
        delta > 1e-12
    } else {
        delta < -1e-12
    };

    // Print label + values
    write!(w, "{label:<20} {val_a:>col_w$}   {val_b:>col_w$}   ")?;

    // Print colored delta
    if no_color {
        writeln!(w, "{delta_str:>12}")?;
    } else if improved {
        w.execute(SetForegroundColor(Color::Green))?;
        write!(w, "{delta_str:>12}")?;
        w.execute(ResetColor)?;
        writeln!(w)?;
    } else if worsened {
        w.execute(SetForegroundColor(Color::Red))?;
        write!(w, "{delta_str:>12}")?;
        w.execute(ResetColor)?;
        writeln!(w)?;
    } else {
        writeln!(w, "{delta_str:>12}")?;
    }

    Ok(())
}

/// Render the batch results as CSV (or JSON).
///
/// All string fields are CSV-quoted, so commas in a filename, model id, or
/// error message do not shift columns.
pub fn print_batch(w: &mut impl Write, results: &[BatchResult], json: bool) -> io::Result<()> {
    if json {
        writeln!(w, "{}", json_pretty(&results)?)?;
        return Ok(());
    }

    writeln!(
        w,
        "file,model,format,tokens,perplexity,mean_logprob,missing_mass,entropy_bias,normalization,bpb,error"
    )?;

    for r in results {
        let model = r.model.as_deref().unwrap_or("");
        let missing_mass = format_optional_f64(r.missing_mass, 4);
        let entropy_bias = r
            .entropy_bias
            .map(|v| format!("{v:+.4}"))
            .unwrap_or_default();
        let bpb = format_optional_f64(r.bpb, 3);
        let error = r.error.as_deref().unwrap_or("");

        writeln!(
            w,
            "{},{},{},{},{:.4},{:.4},{},{},{},{},{}",
            csv_field(&r.file),
            csv_field(model),
            csv_field(&r.format),
            r.tokens,
            r.perplexity,
            r.mean_logprob,
            missing_mass,
            entropy_bias,
            csv_field(&r.normalization),
            bpb,
            csv_field(error),
        )?;
    }

    Ok(())
}

fn format_optional_f64(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:.precision$}"),
        None => String::new(),
    }
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

fn print_legend(w: &mut impl Write) -> io::Result<()> {
    let labels = [
        (Color::Green, "> -0.1"),
        (Color::DarkGreen, "-0.1 to -0.5"),
        (Color::Yellow, "-0.5 to -1.0"),
        (Color::DarkYellow, "-1.0 to -2.0"),
        (Color::Red, "-2.0 to -5.0"),
        (Color::DarkRed, "< -5.0"),
    ];
    write!(w, "Legend: ")?;
    for (color, label) in &labels {
        w.execute(SetForegroundColor(*color))?;
        w.execute(Print(format!("{label}  ")))?;
    }
    w.execute(ResetColor)?;
    writeln!(w)?;
    Ok(())
}

/// Render a token for fixed-width display: control characters become glyphs and
/// the string is truncated with an ellipsis if it exceeds `max_len` characters.
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
        // Guard against underflow if max_len < 3; fall back to a hard cut.
        let keep = max_len.saturating_sub(3);
        let truncated: String = display.chars().take(keep).collect();
        format!("{truncated}...")
    } else {
        display
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render<F>(f: F) -> String
    where
        F: FnOnce(&mut Vec<u8>) -> io::Result<()>,
    {
        let mut buf = Vec::new();
        f(&mut buf).expect("render should succeed");
        String::from_utf8(buf).expect("output should be valid UTF-8")
    }

    #[test]
    fn summary_plain_renders_fields() {
        let summary = SequenceSummary {
            token_count: 3,
            mean_logprob: -0.5,
            total_logprob: -1.5,
            perplexity: 1.65,
            assumed_normalized: false,
            mean_missing_mass: Some(0.1),
        };
        let out = render(|w| print_summary(w, &summary, false));
        assert!(out.contains("Tokens:           3"));
        assert!(out.contains("Perplexity:       1.65"));
        assert!(out.contains("Mean missing mass: 0.1000"));
    }

    #[test]
    fn summary_json_is_parseable() {
        let summary = SequenceSummary {
            token_count: 1,
            mean_logprob: -0.2,
            total_logprob: -0.2,
            perplexity: 1.22,
            assumed_normalized: false,
            mean_missing_mass: None,
        };
        let out = render(|w| print_summary(w, &summary, true));
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["token_count"], 1);
    }

    #[test]
    fn batch_csv_quotes_fields_with_commas() {
        let results = vec![BatchResult {
            file: "a,b.json".to_string(),
            model: Some("vendor,model-1".to_string()),
            format: "openai".to_string(),
            tokens: 2,
            perplexity: 1.5,
            mean_logprob: -0.4,
            missing_mass: Some(0.01),
            entropy_bias: Some(0.02),
            normalization: "pass".to_string(),
            bpb: Some(0.5),
            error: None,
        }];
        let out = render(|w| print_batch(w, &results, false));
        // Header plus one data row, both 11 columns.
        assert!(out.contains("\"a,b.json\""));
        assert!(out.contains("\"vendor,model-1\""));
        let header = out.lines().next().unwrap();
        assert_eq!(header.split(',').count(), 11);
    }

    #[test]
    fn truncate_token_handles_small_max_len() {
        // Must not panic even when max_len < 3.
        let _ = truncate_token("abcdef", 2);
        assert_eq!(truncate_token("abc", 10), "abc");
    }
}
