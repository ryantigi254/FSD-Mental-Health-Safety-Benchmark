from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliable_clinical_benchmark.pipelines import study_b as study_b_pipeline
from reliable_clinical_benchmark.pipelines.study_b import run_study_b


class _UnusedModel:
    def generate(self, *args, **kwargs):
        raise AssertionError("Model should not be called in from_cache mode")

    def chat(self, *args, **kwargs):
        raise AssertionError("Model should not be called in from_cache mode")


@pytest.mark.unit
def test_h_ev_sampling_is_deterministic_and_buffered(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "openr1_psy_splits"
    data_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(1, 61):
        samples.append(
            {
                "id": f"b_{i:03d}",
                "prompt": f"Case {i}",
                "gold_answer": "dummy",
                "incorrect_opinion": "dummy",
                "metadata": {"persona_id": "p1"},
            }
        )

    (data_dir / "study_b_test.json").write_text(
        json.dumps({"samples": samples}), encoding="utf-8"
    )

    cache_path = tmp_path / "study_b_generations.jsonl"
    rows = []
    for s in samples:
        sid = s["id"]
        rows.append(
            {
                "id": sid,
                "variant": "control",
                "prompt": s["prompt"],
                "response_text": f"control response for {sid}",
                "status": "ok",
                "timestamp": "2020-01-01T00:00:00Z",
                "run_id": "test",
                "model_name": "dummy",
                "gold_answer": s["gold_answer"],
                "incorrect_opinion": s["incorrect_opinion"],
            }
        )
        rows.append(
            {
                "id": sid,
                "variant": "injected",
                "prompt": s["prompt"],
                "response_text": f"injected response for {sid}",
                "status": "ok",
                "timestamp": "2020-01-01T00:00:00Z",
                "run_id": "test",
                "model_name": "dummy",
                "gold_answer": s["gold_answer"],
                "incorrect_opinion": s["incorrect_opinion"],
            }
        )

    cache_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    attempted_ids = []

    def _fake_nli_model():
        return object()

    def _fake_h_ev_score(source: str, response: str, nli_model) -> float:
        sid = str(response).rsplit(" ", 1)[-1]
        attempted_ids.append(sid)
        if int(sid.split("_")[1]) <= 9:
            raise RuntimeError("forced failure")
        return 0.25

    monkeypatch.setattr(study_b_pipeline, "NLIModel", _fake_nli_model)
    monkeypatch.setattr(study_b_pipeline, "evidence_hallucination_score", _fake_h_ev_score)

    output_dir = tmp_path / "results"

    run_study_b(
        model=_UnusedModel(),
        data_dir=str(data_dir),
        max_samples=60,
        output_dir=str(output_dir),
        model_name="dummy",
        use_nli=True,
        from_cache=str(cache_path),
    )

    expected_attempted = [f"b_{i:03d}" for i in range(1, 60)]
    assert attempted_ids == expected_attempted

    result_path = output_dir / "dummy" / "study_b_results.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["evidence_hallucination_n_attempted"] == 59
    assert result["evidence_hallucination_n_scored"] == 50
    assert result["evidence_hallucination"] == pytest.approx(0.25, abs=1e-6)
