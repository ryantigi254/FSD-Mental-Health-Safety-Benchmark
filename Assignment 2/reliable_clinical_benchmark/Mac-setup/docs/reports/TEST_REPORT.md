# Test Report – Mac setup

## Latest recorded run (local venv)

Command:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -q -o addopts=
```

Observed output:

```text
.......s........s.                                    [100%]
===================== warnings summary ======================
.../typer/completion.py:122: DeprecationWarning: 'BaseCommand' is deprecated and will be removed in Click 9.0. Use 'Command' instead.
.../spacy/cli/_util.py:23: DeprecationWarning: Importing 'parser.split_arg_string' is deprecated; it will only be available in 'shell_completion' in Click 9.0.
.../spacy/language.py:2141: FutureWarning: Possible set union at position 6328
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
16 passed, 2 skipped, 3 warnings in 4.78s
```

- Result: all tests in `tests/unit` and `tests/integration` pass on the Mac virtual environment.
- Skips: 2 tests are `pytest.skip` guarded (e.g., when optional NLI model is unavailable).

## Recommended verbose run

For streamed progress:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -vv -s
```

To capture a log while streaming:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -vv -s | tee tests/test_run_$(date +%Y%m%d_%H%M%S).log
```

Generates a timestamped log under `tests/` suitable for sharing as an artefact.
