# 🍰 CAKEbench — Carteakey's Assessment of Knowledge & Execution

A personal LLM benchmark for evaluating small open-source reasoning models on challenging, hand-crafted prompts across programming, data science, ML, and (future) agentic tasks.

## Philosophy

- **Home-grown prompts** — no prompt contamination from public benchmarks
- **Multi-domain** — tests programming, algorithms, ML, visual coding, debugging
- **Reasoning-aware** — test the same model at different reasoning effort levels (low/med/high)
- **Practical** — measures what matters: does the code run? Is it correct? How fast?

## Models Under Test

| Model | Params | Reasoning Variants | Quant |
|---|---|---|---|
| GPT-OSS-120B | 120B | Low / Med / High | MXFP4 |
| GPT-OSS-20B | 20B | Low / Med / High | MXFP4 |
| Qwen3-Coder-Next | MoE | N/A (non-reasoning) | MXFP4 |

## Prompt Sets (Domains)

| Domain | Prompts | Eval Type | What it Tests |
|---|---|---|---|
| 🎨 Vibe Coding | 5 | Human / LLM-judge | Visual quality, physics, interactivity |
| 🧮 Algorithms | 9 | Auto (run + assert) | DP, greedy, optimization correctness |
| 🐛 Debugging | 1+ | Auto (run + compare) | Code comprehension, bug fixing |
| 🤖 ML From Scratch | 1+ | Auto (accuracy check) | ML fundamentals without libraries |
| 🔮 Agentic (future) | TBD | Auto + Human | Tool use, multi-step reasoning |

## Scoring System

### Per-Prompt Scoring (0–100)

#### Auto-eval prompts (algorithms, debugging, ML)

| Dimension | Weight | Scoring |
|---|---|---|
| **Runs** | 30% | `python output.py` exits 0 → 30, crash → 0 |
| **Correct output** | 50% | stdout matches expected values → 50, partial → 25, wrong → 0 |
| **Code quality** | 10% | No syntax warnings, clean structure → 10 |
| **Efficiency** | 10% | Completes within timeout → 10, slow → 5 |

#### Human-eval prompts (vibe coding)

| Dimension | Weight | Scoring |
|---|---|---|
| **Runs** | 20% | Window opens, no crash → 20 |
| **Visual quality** | 30% | Aesthetics, colors, layout (1–10 × 3) |
| **Correctness** | 30% | Follows spec (number of balls, colors, physics) |
| **Interactivity** | 20% | Responsive, smooth animations |

### Aggregate CAKE Score

```
CAKE Score = weighted_avg(
    Algorithms:     35%   (9 prompts)
    Vibe Coding:    25%   (5 prompts)
    Debugging:      20%   (1+ prompts)
    ML:             20%   (1+ prompts)
)
```

### Leaderboard Output

```
╔═══════════════════════════════════════════════════════════════╗
║                     🍰 CAKEbench v0.1                        ║
╠══════════════════╦════════╦═══════╦═══════╦═════╦════════════╣
║ Model            ║ CAKE   ║ Algos ║ Vibe  ║ Debug ║ ML       ║
╠══════════════════╬════════╬═══════╬═══════╬═════╬════════════╣
║ GPT-OSS-120B-Hi  ║  87.2  ║  92   ║  85   ║ 80  ║   90     ║
║ GPT-OSS-120B-Med ║  74.5  ║  80   ║  72   ║ 68  ║   75     ║
║ GPT-OSS-120B-Low ║  58.1  ║  65   ║  55   ║ 50  ║   60     ║
║ GPT-OSS-20B-Hi   ║  71.0  ║  78   ║  65   ║ 65  ║   73     ║
║ Qwen3-Coder-Next ║  82.4  ║  88   ║  80   ║ 75  ║   85     ║
╚══════════════════╩════════╩═══════╩═══════╩═════╩════════════╝
```

## Key Insights to Capture

1. **Reasoning scaling** — does Low→Med→High actually improve output quality?
2. **Size vs. reasoning** — GPT-OSS-20B-High vs GPT-OSS-120B-Low: can more thinking compensate for fewer params?
3. **Speed-quality tradeoff** — tk/s vs CAKE score scatter plot
4. **Domain strengths** — which models excel at visual vs algorithmic tasks?

## Implementation Phases

### Phase 1 (Current) ✅
- [x] Framework: config, orchestration, runner, results, plots
- [x] Prompt sets: vibe coding (5), algorithms (9), debugging (1), ML (1)
- [x] Model configs: GPT-OSS-120B (3 variants), GPT-OSS-20B (3 variants), Qwen3

### Phase 2 (Next)
- [ ] Auto-scorer: run output.py, assert expected values, compute per-prompt score
- [ ] CAKE score aggregation across prompt sets
- [ ] Leaderboard generator (markdown + image)

### Phase 3 (Future)
- [ ] LLM-as-judge for vibe coding (send screenshots to vision model)
- [ ] Agentic prompt set (tool use, multi-step)
- [ ] Historical tracking (compare across runs)
- [ ] Reasoning token analysis (thinking tokens vs output quality)
