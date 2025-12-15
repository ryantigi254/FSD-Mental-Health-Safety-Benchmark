"""Integration smoke test: load Psyche-R1 locally and run a tiny generation.

This is intentionally opt-in because it requires local weights and GPU.

Enable by setting:
  - PSYCHE_R1_INTEGRATION=1
"""

import os
import re
from pathlib import Path

import pytest

from reliable_clinical_benchmark.models.base import GenerationConfig
from reliable_clinical_benchmark.models.psyche_r1_local import PsycheR1LocalRunner


@pytest.mark.integration
@pytest.mark.slow
def test_psyche_r1_local_can_generate_and_think_block_is_extractable():
    if os.environ.get("PSYCHE_R1_INTEGRATION", "").strip() != "1":
        pytest.skip("Set PSYCHE_R1_INTEGRATION=1 to run this test.")

    model_dir = Path("models/Psyche-R1")
    if not model_dir.exists():
        pytest.skip("Local weights not found at models/Psyche-R1")

    runner = PsycheR1LocalRunner(
        model_name=str(model_dir),
        config=GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=96,
        ),
    )

    prompt = "Give a brief clinical differential for persistent low mood."

    # Normalised output for Study A parsing.
    out = runner.generate(prompt, mode="cot")
    assert isinstance(out, str)
    assert "REASONING:" in out
    assert "DIAGNOSIS:" in out

    # Raw generation (no normalisation) so we can detect <think> tags if present.
    encoded = runner._build_inputs(prompt, mode="cot")
    input_token_count = int(encoded["input_ids"].shape[-1])
    gen = runner.model.generate(
        **encoded,
        max_new_tokens=96,
        do_sample=True,
        temperature=runner.config.temperature,
        top_p=runner.config.top_p,
        top_k=20,
        repetition_penalty=1.05,
        eos_token_id=runner.tokenizer.eos_token_id,
        pad_token_id=runner.tokenizer.pad_token_id,
    )
    generated_only = gen[0, input_token_count:]
    raw = runner.tokenizer.decode(generated_only, skip_special_tokens=True).strip()
    assert raw

    # If the model followed the instruction to wrap reasoning, prove we can extract it.
    if "<think>" in raw and "</think>" in raw:
        m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        assert m is not None
        assert m.group(1).strip()
        after = raw[m.end() :].strip()
        assert after


