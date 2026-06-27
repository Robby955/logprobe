//! End-to-end tests that drive the compiled `logprobe` binary.
//!
//! Cargo sets `CARGO_BIN_EXE_logprobe` for integration tests, so no extra
//! dependency is needed to locate and run the binary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_logprobe"))
}

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
}

#[test]
fn summary_from_file_argument() {
    let out = bin()
        .arg("summary")
        .arg(fixture("demo/gpt4o_mini_creative.json"))
        .output()
        .expect("run logprobe");
    assert!(out.status.success(), "summary should exit 0");
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("=== Summary ==="));
    assert!(stdout.contains("Tokens:"));
}

#[test]
fn summary_from_stdin() {
    let mut child = bin()
        .arg("summary")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn logprobe");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"[{"token":"Hi","logprob":-0.5},{"token":"!","logprob":-1.0}]"#)
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("Tokens:           2"));
}

#[test]
fn compare_preserves_argument_order_in_json() {
    let a = fixture("demo/gpt4o_mini_creative.json");
    let b = fixture("demo/gpt4o_mini_code.json");
    let out = bin()
        .arg("compare")
        .arg(&a)
        .arg(&b)
        .arg("--json")
        .output()
        .expect("run logprobe compare");
    assert!(out.status.success(), "compare should exit 0");

    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    // First positional becomes file_a, second becomes file_b.
    assert_eq!(report["file_a"]["label"], a.to_string_lossy().as_ref());
    assert_eq!(report["file_b"]["label"], b.to_string_lossy().as_ref());
}

#[test]
fn bpb_without_bytes_exits_nonzero_in_text_mode() {
    // The JSONL fixture has no byte arrays, so strict BPB must refuse.
    let out = bin()
        .arg("bpb")
        .arg(fixture("tests/fixtures/stream.jsonl"))
        .output()
        .expect("run logprobe bpb");
    assert!(
        !out.status.success(),
        "BPB without bytes should exit non-zero in text mode"
    );
}

#[test]
fn bpb_without_bytes_json_mode_exits_zero_with_error_status() {
    let out = bin()
        .arg("bpb")
        .arg(fixture("tests/fixtures/stream.jsonl"))
        .arg("--json")
        .output()
        .expect("run logprobe bpb --json");
    assert!(out.status.success(), "JSON mode should still exit 0");
    let parsed: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(parsed["status"], "error");
}
