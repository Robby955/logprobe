# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-27

### Fixed

- **Gemini BPB correctness.** The Gemini parser no longer fabricates per-token
  byte arrays from the token strings (`token.as_bytes()`). Gemini does not
  return byte arrays, and synthesizing them is exactly the fallback the
  strict-BPB path refuses elsewhere. `bpb` now correctly reports unavailable
  for Gemini input instead of returning a confidently-wrong value.
- **Entropy documentation.** Corrected the false claim that the normalized
  (renormalized top-k) entropy universally underestimates the true entropy. Its
  bias relative to the true Shannon entropy has indeterminate sign. The partial
  top-k entropy remains an unconditional lower bound, and the docs now state the
  bound without an unnecessary qualifier.
- **CSV output.** `batch` CSV now quotes and escapes every string field
  (file, model, format, normalization, error), so a comma in a model id or
  filename no longer shifts downstream columns. The header now always includes
  the `error` column, keeping every row at a fixed width.
- **`truncate_token` underflow.** Guarded the display-truncation helper against
  a panic when called with a maximum length below 3.

### Changed

- **Typed library errors.** `parse::parse_input` and `parse::parse_string`
  now return `Result<_, parse::ParseError>` (a `thiserror` enum) instead of
  `anyhow::Result`, so library consumers can match on the failure mode and no
  longer inherit an `anyhow` dependency. `anyhow` is confined to the binary.
- **Writable, testable output.** Every `output::print_*` function is now generic
  over its writer (`&mut impl Write`) and returns `io::Result<()>`, so output
  can be redirected or captured and the renderers are unit-tested directly.
- Single-sourced the missing-mass unreliability threshold so `diagnose` and the
  per-token entropy metrics report the same cutoff.
- CLI `--format` help and examples now mention the Gemini and Ollama formats,
  which were already supported.

### Added

- Crate-level and `math::missing_mass` runnable doctests.
- `#[must_use]` on the pure math, metrics, diagnostics, and filter functions.
- CLI integration tests (`tests/cli.rs`) covering stdin, file input, compare
  argument ordering, and BPB exit codes; plus output-rendering and Gemini
  BPB-refusal unit tests.

### Packaging

- Added `documentation` metadata and an `exclude` list so the published tarball
  omits `research/` (~10 MB) and `docs/` while keeping the `demo/` fixtures the
  test suite depends on.
- Added this changelog.

## [0.3.0]

- Initial public release: parsing for OpenAI, vLLM, Gemini, Ollama, and JSONL
  logprob formats; `diagnose`, `validate`, `summary`, `entropy`, `confidence`,
  `bpb`, `highlight`, `compare`, and `batch` subcommands; strict bits-per-byte
  that refuses to guess byte counts.
