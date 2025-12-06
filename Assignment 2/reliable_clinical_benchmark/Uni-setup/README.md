# Mental Health LLM Safety Benchmark

Publication-ready evaluation framework for assessing reasoning models on faithfulness, sycophancy, and longitudinal drift in mental health contexts.

## Quick Start

### 1. Environment Setup

```bash
cd Assignment\ 2/Uni-setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env and add:
# HUGGINGFACE_API_KEY=your_key_here
```

### 3. Setup LM Studio (for PsyLLM)

See `docs/ENVIRONMENT.md` for detailed instructions on setting up LM Studio for local PsyLLM inference.

### 4. Prepare Data

Ensure your data files are in place:

- `data/openr1_psy_splits/study_a_test.json` - Study A faithfulness data
- `data/openr1_psy_splits/study_b_test.json` - Study B sycophancy data
- `data/openr1_psy_splits/study_c_test.json` - Study C longitudinal drift data
- `data/adversarial_bias/biased_vignettes.json` - Adversarial bias cases

### 5. Run Tests

```bash
pytest tests/unit/ -v --cov=src
```

### 6. Run Evaluation

```bash
# Run a single study
python scripts/run_evaluation.py --model psyllm --study A

# Run all studies
python scripts/run_evaluation.py --model psyllm --study all

# With custom parameters
python scripts/run_evaluation.py --model qwq --study B --max-samples 10 --temperature 0.7
```

## Single Environment Approach

This project uses **one virtual environment** for all three studies to ensure:

- Reproducibility across all experiments
- Consistent dependency versions
- Standard publication practices

## Evaluation Studies

### Study A: Faithfulness (403 prompts/model)

- **Faithfulness Gap (Δ)**: Measures if reasoning is functional or decorative
- **Step-F1**: Validates reasoning content quality against gold standards
- **Silent Bias Rate**: Detects hidden demographic biases

### Study B: Sycophancy (1,035 prompts/model)

- **Sycophancy Probability (P_Syc)**: Measures agreement shift under user pressure
- **Flip Rate**: Clinical failure rate (correct → incorrect transitions)
- **Evidence Hallucination**: Detects fabricated symptoms to support user's incorrect opinion
- **Turn of Flip (ToF)**: Defines safe conversation window

### Study C: Longitudinal Drift (460 prompts/model)

- **Entity Recall Decay**: Measures forgetting of critical entities over turns
- **Knowledge Conflict Rate**: Detects self-contradictions
- **Continuity Score**: Measures adherence to treatment plan

## Models Evaluated

1. **PsyLLM-8B** (LM Studio, local) – primary path
2. **GPT-OSS-120B** (LM Studio GGUF MXFP4, `lmstudio-community/openai-gpt-oss-120b-gguf-mxfp4`)
3. **GPT-OSS-20B** (LM Studio GGUF, record quant e.g., MXFP4/FP16)
4. **QwQ-32B** (LM Studio GGUF FP16)
5. **DeepSeek-R1-Distill-Llama-70B** (LM Studio GGUF Q6_K, https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
6. **Qwen3-8B** (Hugging Face API baseline)

Current LM Studio runs use these quantised GGUF builds (as downloaded in LM Studio). Remote API runners stay available for QwQ/DeepSeek/GPT-OSS if you prefer hosted inference. All LM Studio communication is handled by `lmstudio_client` for consistent error handling.

## Project Structure

```
Uni-setup/
├── src/
│   └── reliable_clinical_benchmark/
│       ├── models/          # Model runners
│       ├── metrics/          # Evaluation metrics
│       ├── pipelines/        # Study pipelines
│       ├── data/             # Data loaders
│       ├── eval/             # Evaluation orchestration
│       └── utils/            # Utilities (NLI, NER, stats)
├── scripts/
│   ├── run_evaluation.py    # Main evaluation script
│   └── update_leaderboard.py # Leaderboard generator
├── tests/
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── notebooks/               # Analysis notebooks (Jupyter)
│   ├── study_a_analysis.ipynb
│   ├── study_b_analysis.ipynb
│   └── study_c_analysis.ipynb
├── docs/                    # Documentation
│   ├── ENVIRONMENT.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── study_a_faithfulness.md
│   ├── study_b_sycophancy.md
│   └── study_c_drift.md
├── data/                    # Test data (frozen splits)
└── results/                 # Evaluation outputs
```

## Example Usage

```bash
# Evaluate PsyLLM on Study A
python scripts/run_evaluation.py --model psyllm --study A

# Evaluate QwQ-32B on all studies
python scripts/run_evaluation.py --model qwq --study all

# Quick test run with limited samples
python scripts/run_evaluation.py --model psyllm --study B --max-samples 5

# Update leaderboard after running evaluations
python scripts/update_leaderboard.py --results-dir results
```

## Documentation

- `docs/ENVIRONMENT.md` - Environment setup and configuration
- `docs/EVALUATION_PROTOCOL.md` - Detailed evaluation procedure
- `docs/study_a_faithfulness.md` - Study A implementation guide (metrics, formulas, design decisions)
- `docs/study_b_sycophancy.md` - Study B implementation guide (metrics, formulas, design decisions)
- `docs/study_c_drift.md` - Study C implementation guide (metrics, formulas, design decisions)

## Analysis Notebooks

After running evaluations, use the Jupyter notebooks in `notebooks/` to analyse results:

- `notebooks/study_a_analysis.ipynb` - Rank models by faithfulness gap, visualise Step-F1, assess silent bias
- `notebooks/study_b_analysis.ipynb` - Compare sycophancy probability, flip rates, evidence hallucination, safe conversation windows
- `notebooks/study_c_analysis.ipynb` - Plot entity recall decay curves, compare drift slopes, assess knowledge conflicts

Each notebook includes:
- Model ranking tables
- Visualisations with error bars (bootstrap CIs)
- Safety card summaries (which models pass thresholds)
- Interpretation guides connecting metrics to clinical deployment decisions

## Citation

```bibtex
@article{gichuru2026mental,
  title={Mental Health LLM Safety Benchmark: Evaluating Reasoning Models on Faithfulness, Sycophancy, and Longitudinal Drift},
  author={Gichuru, Ryan Mutiga},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License

