# Test Report – Mac-setup

## Latest recorded run (from `.mh-llm-benchmark-env`)

Command used:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -q -o addopts=
```

Observed output:

```text
.......s........s.                                    [100%]
===================== warnings summary ======================
.mh-llm-benchmark-env/lib/python3.11/site-packages/typer/completion.py:122
  .../typer/completion.py:122: DeprecationWarning: 'BaseCommand' is deprecated and will be removed in Click 9.0. Use 'Command' instead.
    cli: click.BaseCommand,

.mh-llm-benchmark-env/lib/python3.11/site-packages/spacy/cli/_util.py:23
  .../spacy/cli/_util.py:23: DeprecationWarning: Importing 'parser.split_arg_string' is deprecated, it will only be available in 'shell_completion' in Click 9.0.
    from click.parser import split_arg_string

tests/integration/test_pipelines.py::test_study_c_pipeline
  .../spacy/language.py:2141: FutureWarning: Possible set union at position 6328
    deserializers["tokenizer"] = lambda p: self.tokenizer.from_disk(  # type: ignore[union-attr]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
16 passed, 2 skipped, 3 warnings in 4.78s
```

- **Result**: all tests in `tests/unit` and `tests/integration` pass on the Mac-setup env.
- **Skips**: 2 tests are `pytest.skip`-guarded (e.g. when optional NLI model is not available), by design.

## Recommended verbose run (streamed progress)

For a progress log that shows each test as it runs (and to integrate with CI if needed), use:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -vv -s
```

- `-vv` prints every test name and its status.
- `-s` allows `print` output from tests to be shown live.

To capture a full log to file whilst still seeing it in the terminal:

```bash
cd "Assignment 2/reliable_clinical_benchmark/Mac-setup"
PYTHONPATH=src python -m pytest -vv -s | tee tests/test_run_$(date +%Y%m%d_%H%M%S).log
```

This produces a timestamped log under `tests/` that can be attached as an appendix or artefact for the dissertation.


