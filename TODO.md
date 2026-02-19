# TODO

Roadmap and task list for quaLLM / CAKEbench.

## 🔥 High Priority

- [ ] **Auto-scorer** — run `output.py`, assert expected values, compute per-prompt score (0–100)
- [ ] **CAKE score aggregation** — weighted average across prompt sets (Algos 35%, Vibe 25%, Debug 20%, ML 20%)
- [ ] **Leaderboard generator** — render markdown + image leaderboard from scored results
- [ ] **Expected outputs** — add expected stdout / assertions to algorithm and debugging prompts

## 🛠️ Medium Priority

- [ ] **LLM-as-judge for vibe coding** — send screenshots to a vision model for automated visual quality scoring
- [ ] **Reasoning token analysis** — extract and compare thinking tokens vs output quality across reasoning levels
- [ ] **Historical tracking** — compare scores across runs over time, detect regressions
- [ ] **HTML report generation** — generate a rich HTML report per run with embedded plots, code, and screenshots
- [ ] **Prompt set: Agentic** — design multi-step, tool-use prompts for agentic evaluation
- [ ] **Parallel model evaluation** — run multiple models concurrently when using `--no-start-server`

## 📦 Code Quality

- [ ] **Add `pyproject.toml`** — proper Python packaging with entry point (`quallm` CLI)
- [ ] **Unit tests** — test config loading, result serialization, blind evaluation logic
- [ ] **CI pipeline** — GitHub Actions for lint + tests on push
- [ ] **Type hints** — add `py.typed` marker and ensure full mypy coverage
- [ ] **Logging cleanup** — configurable log level via `--verbose` / `--quiet` flags

## 💡 Ideas / Future

- [ ] Support non-llama.cpp backends (vLLM, Ollama, OpenAI-compatible APIs)
- [ ] Web UI dashboard for browsing results and scoring vibe-coding outputs
- [ ] Prompt difficulty calibration — tag prompts as easy/medium/hard and weight accordingly
- [ ] Model cost tracking — estimate $/run based on token counts and hardware utilization
- [ ] Export results to CSV / JSON-lines for external analysis
