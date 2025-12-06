# Test Report – Mac setup

Latest run (Mac venv): `docs/reports/test_logs/mac_pytest_20251206_171548.log`

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
source .mh-llm-benchmark-env/bin/activate
PYTHONPATH=src python -m pytest -q | tee docs/reports/test_logs/mac_pytest_20251206_171548.log
```

Key results (12.96s total):
- Outcome: 26 passed, 0 skipped, 0 failed.
- Warnings: Click `BaseCommand` deprecation, Click `parser.split_arg_string` deprecation, spaCy tokenizer future warning (see log).
- Coverage: 53% overall (HTML in `htmlcov`). High-miss areas: `metrics/faithfulness.py`, `models/lmstudio_client.py`, `models/psyllm.py`, `eval/*`, `pipelines/*`, `utils/stats.py`.

Next steps:
- Leave the log in `docs/reports/test_logs/` as the artefact for this Mac run.
- Re-run after substantial changes with the same command; adjust `--maxfail` or `-vv` only if troubleshooting.
