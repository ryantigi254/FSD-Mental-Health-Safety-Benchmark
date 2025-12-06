# Evaluation Protocol (Mac)

Mac-only notes for running and interpreting the benchmark locally. Full evaluation on larger models remains on the Uni setup.

## Status (Mac)
- No full metric run has been completed on Mac yet. Current artefact: `results/qwen3-8b-mlx/study_a_generations.jsonl` (generation-only; produced while LM Studio was offline, so rerun with a live server).
- Use this document when launching Mac smoke tests or small subset runs only.

## Data (frozen)
- Study A: `data/openr1_psy_splits/study_a_test.json` (300 samples; no persona metadata). Bias stressors: `data/adversarial_bias/biased_vignettes.json` (58 items).
- Study B: `data/openr1_psy_splits/study_b_test.json` – 40 single-turn items (4 each across 10 personas) and 10 multi-turn cases (one per persona).
- Study C: `data/openr1_psy_splits/study_c_test.json` – 30 cases (3 per persona, 10 personas).
- Personas source: `../Uni-setup/docs/personas/persona_registry_v2.json`.

## Protocol snapshots
- **Study A (Faithfulness):** Faithfulness Gap (Δ = Acc_CoT − Acc_Early), Step-F1 (token overlap 0.6), Silent Bias Rate on adversarial items.
- **Study B (Sycophancy):** P_Syc (agreement shift), Flip Rate (Correct→Incorrect), Evidence Hallucination via NLI, Turn of Flip on multi-turn pressure.
- **Study C (Longitudinal Drift):** Entity Recall over 10 turns, Knowledge Conflict rate via NLI, Continuity Score with MiniLM embeddings.

## Running on Mac
```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate

# Keep runs small on Mac
PYTHONPATH=src python scripts/run_evaluation.py --model qwen3-8b-mlx --study A --max-samples 5
PYTHONPATH=src python scripts/run_evaluation.py --model PsyLLM-8B --study B --max-samples 5
```

## Outputs and next actions
- Expected outputs (once run): `results/<model>/study_a_results.json`, `study_b_results.json`, `study_c_results.json`.
- Leaderboard generation (`python scripts/update_leaderboard.py`) is blocked until at least one Mac metric run completes.
- Before running: ensure LM Studio local server is live for Qwen3/PsyLLM, and `HUGGINGFACE_API_KEY` is set if using the Hugging Face Inference API path.

