## `src/` layout (Uni-setup)

The canonical Python package is:

- `src/reliable_clinical_benchmark/`

Everything under that package uses absolute imports like `reliable_clinical_benchmark.models...`.

### Package Structure

```
src/reliable_clinical_benchmark/
├── models/          # Model runners (LM Studio, local HF, remote API)
│   ├── base.py      # ModelRunner abstract base class (generate, chat, generate_with_reasoning)
│   ├── factory.py   # Model factory for getting runners by ID
│   └── ...          # Model-specific implementations
├── pipelines/       # Evaluation pipelines
│   ├── study_a.py   # Faithfulness evaluation (CoT vs Direct, Silent Bias)
│   ├── study_b.py   # Sycophancy evaluation (single-turn + multi-turn)
│   └── study_c.py   # Longitudinal drift evaluation
├── metrics/         # Metric calculation functions
│   ├── faithfulness.py
│   ├── sycophancy.py
│   └── drift.py
├── data/            # Data loaders for each study
│   ├── study_a_loader.py
│   ├── study_b_loader.py
│   └── study_c_loader.py
├── eval/            # Evaluation utilities
│   ├── runtime_checks.py  # Schema validation
│   └── results_schema.py
└── utils/           # Shared utilities (NLI, NER, stats, logging)
```

### ModelRunner Interface

All model runners inherit from `ModelRunner` (in `models/base.py`) and implement:

- `generate(prompt: str, mode: str) -> str`: Single-turn text generation
- `chat(messages: List[Dict[str, str]], mode: str) -> str`: Multi-turn conversation with rolling context
- `generate_with_reasoning(prompt: str) -> Tuple[str, str]`: Extract reasoning + answer

The `chat()` method enables proper multi-turn conversations:
- Accepts structured message history with roles (system/user/assistant)
- Maintains rolling context across turns
- Uses chat templates (transformers) or chat completion APIs (LM Studio)

### Study A Architecture

Study A evaluates **Faithfulness** (whether model reasoning drives predictions) through two generation modes:

1. **Main Generation** (`run_study_a`):
   - **CoT mode** (`mode="cot"`): Chain-of-Thought reasoning with step-by-step analysis
   - **Direct mode** (`mode="direct"`): Immediate answer without explicit reasoning
   - Each sample generates both modes (2 generations per sample)
   - Metrics: Faithfulness gap (Δ), accuracy (CoT vs Direct), step F1

2. **Bias Evaluation** (separate workflow):
   - **Adversarial bias cases**: Tests Silent Bias Rate (R_SB)
   - Uses CoT mode only (reasoning required to detect "silence")
   - Separate generation script: `hf-local-scripts/run_study_a_bias_generate_only.py`
   - Output: `results/{model-id}/study_a_bias_generations.jsonl`
   - Metric: Silent Bias Rate (measures if model hides demographic bias in reasoning)

**Workflow**:
- Main generations: `study_a_generations.jsonl` (CoT + Direct per sample)
- Bias generations: `study_a_bias_generations.jsonl` (CoT only, adversarial cases)
- Metrics calculated separately and merged in final results

### Study B Architecture

Study B is split into two distinct generation modes:

1. **Single-Turn** (`_generate_single_turn_study_b`):
   - Control + injected variants per sample
   - Metrics: Sycophancy probability, flip rate, evidence hallucination

2. **Multi-Turn** (`_generate_multi_turn_study_b`):
   - Iterative generation with rolling context
   - Each turn is a separate API call
   - Full conversation history passed via `model.chat()`
   - Metric: Turn of Flip (ToF)

### Study C Architecture

Study C evaluates **Longitudinal Drift** (consistency over multi-turn conversations) through two variants per turn:

1. **Summary Variant** (`variant="summary"`):
   - Accumulated context string (patient summary + all turns up to current)
   - Uses `model.generate()` with `mode="summary"`
   - Metric: Entity Recall Decay (measures information retention over turns)

2. **Dialogue Variant** (`variant="dialogue"`):
   - Conversation history built turn-by-turn
   - Currently uses string concatenation (could be migrated to `model.chat()`)
   - Metric: Knowledge Conflict Rate (measures contradictions across turns)

**Multi-Turn Structure**:
- 10 turns per case
- Each turn generates both variants (summary + dialogue)
- Total: 2 generations × 10 turns = 20 generations per case
- Rolling context: Each turn sees all previous conversation

### Notes

- `__pycache__/` directories are runtime artefacts created by Python and should be ignored.
- All imports use absolute paths: `from reliable_clinical_benchmark.models.base import ModelRunner`
- For test structure, see `tests/README.md` (pytest unit tests) and `docs/studies/TESTING_GUIDE.md` (smoke tests)

