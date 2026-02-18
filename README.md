# quaLLM 🧪

**Qua**litative **LLM** evaluation — a lightweight framework for testing open-source models side-by-side.

Built for vibe-coding benchmarks, visual quality checks, and performance comparisons of small OSS models served via [llama.cpp](https://github.com/ggml-org/llama.cpp).

## Features

- **Multi-model evaluation** — run the same prompt against multiple models sequentially
- **Server orchestration** — auto-start/stop `llama-server` instances with per-model configs
- **Performance benchmarking** — captures prompt processing speed, generation speed (tk/s), token counts
- **Structured results** — every run saved with metadata, raw output, extracted code, and perf metrics  
- **Comparison plots** — dark-themed bar charts for gen speed, prompt speed, token counts
- **Blind evaluation** — anonymize model identities for unbiased human scoring (`--blind`)
- **Screenshot capture** — auto-run generated code and capture window screenshots (`--run-outputs`)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Dry run (validate everything, generate dummy plots)
python run_eval.py \
    --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \
    --prompt prompts/bouncing_balls.txt \
    --dry-run

# Real run (servers already running)
python run_eval.py \
    --models model_configs/qwen3_coder_next.yaml \
    --prompt prompts/bouncing_balls.txt \
    --no-start-server

# Full orchestration (start → eval → stop, sequentially)
python run_eval.py \
    --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \
    --prompt prompts/bouncing_balls.txt

# Blind evaluation with auto-screenshots
python run_eval.py \
    --models model_configs/gpt_oss_120b.yaml model_configs/qwen3_coder_next.yaml \
    --prompt prompts/bouncing_balls.txt \
    --blind --run-outputs --display-seconds 8
```

## Results Structure

Each run creates a timestamped directory:

```
results/bouncing_balls_20260218_153306/
├── metadata.json           # Run metadata
├── comparison.png          # Performance comparison chart
├── summary_table.png       # Summary table image
├── blind_key.json          # Model identity mapping (only with --blind)
├── gpt_oss_120b/
│   ├── output.txt          # Raw model output
│   ├── output.py           # Extracted Python code
│   ├── perf.json           # Performance metrics
│   └── screenshot.png      # Window capture (only with --run-outputs)
└── qwen3_coder_next/
    ├── output.txt
    ├── output.py
    ├── perf.json
    └── screenshot.png
```

## Adding Models & Prompts

**New model** — create a YAML file in `model_configs/`:
```yaml
name: "My-Model"
alias: "org/model-name"
model_path: "/path/to/model.gguf"
port: 8001
server_args:
  fit: "on"
  threads: 10
  flash-attn: "on"
```

**New prompt** — drop a `.txt` file in `prompts/`:
```
Write a Python program that simulates a double pendulum...
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--models` / `-m` | Model config YAML paths (one or more) |
| `--prompt` / `-p` | Prompt file path (.txt or .yaml) |
| `--no-start-server` | Skip server orchestration (servers already running) |
| `--dry-run` | Validate configs, generate dummy results |
| `--blind` | Anonymize model names for unbiased evaluation |
| `--run-outputs` | Run generated code + capture screenshots |
| `--display-seconds` | Time to wait before capturing screenshot (default: 5) |
| `--output-dir` / `-o` | Custom results directory |
| `--max-tokens` | Max generation tokens (default: 16384) |

## Dependencies

- Python 3.10+
- `requests`, `pyyaml`, `matplotlib`
- Optional: `xdotool` + `imagemagick` (for `--run-outputs` screenshot capture)
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` binary
