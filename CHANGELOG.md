# Changelog

All notable changes to quaLLM are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-02-18

### Added
- **CAKEbench** — personal LLM benchmark with scoring system, leaderboard spec, and multi-domain prompt evaluation (`CAKEBENCH.md`)
- GPT-OSS reasoning variants — `Low`, `Med`, `High` reasoning-level configs for both GPT-OSS-120B and GPT-OSS-20B
- Runtime error detection — `output.py` crashes now logged to `runtime_error.txt` with exit code and stderr
- Increased evaluation timeout to 30 minutes for long-running prompts

### Fixed
- Multi-fallback screenshot capture — tries xdotool → scrot → PIL in order, with proper error handling at each step

## [0.1.0] — 2026-02-18

### Added
- **Core evaluation framework** — `quallm` package with `config`, `orchestrator`, `runner`, `results`, and `plotting` modules
- **Prompt sets** — YAML-based prompt set definitions for batch evaluation across domains:
  - Algorithms (9 prompts), Vibe Coding (5), Debugging (1), ML From Scratch (1)
- **Curated prompts** — 16 hand-crafted prompts moved from inbox to `prompts/`
- **Model orchestration** — auto start/stop `llama-server` instances per model config with health checks
- **Performance capture** — prompt processing speed, generation speed (tk/s), token counts per model run
- **Structured results** — timestamped directories with metadata, raw output, extracted code, and perf metrics
- **Comparison plots** — dark-themed bar charts and summary tables via matplotlib
- **Blind evaluation** — `--blind` flag to anonymize model identities for unbiased human scoring
- **Screenshot capture** — `--run-outputs` flag to auto-run generated code and capture window screenshots
- **Dry-run mode** — `--dry-run` to validate configs and generate dummy results without running models
- **CLI** — full argument parser with `--models`, `--prompt`, `--prompt-set`, `--max-tokens`, `--output-dir`, etc.
- Initial model configs: `gpt_oss_120b.yaml`, `qwen3_coder_next.yaml`
- Shell helpers: `bench-llama-cpp.sh`, `bench-sweep-llama-cpp.sh`, `run-llama-cpp-gpt-oss.sh`, `run-llama-cpp-qwen3.sh`
