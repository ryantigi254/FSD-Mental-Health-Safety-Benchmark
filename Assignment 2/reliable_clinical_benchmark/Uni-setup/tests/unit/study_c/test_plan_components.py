from __future__ import annotations

from typing import List, Tuple

import pytest

from reliable_clinical_benchmark.utils.plan_components import (
    PlanComponent,
    classify_plan_components,
    render_plan_from_components,
    extract_recommendation_candidates,
    nli_filter_candidates,
)


class _StubNLI:
    def __init__(self, entail_hypotheses_containing: List[str]):
        self._needles = [n.lower() for n in entail_hypotheses_containing]

    def batch_predict(self, premise_hypothesis_pairs: List[Tuple[str, str]]):
        verdicts = []
        for _premise, hypothesis in premise_hypothesis_pairs:
            h = str(hypothesis).lower()
            if any(needle in h for needle in self._needles):
                verdicts.append("entailment")
            else:
                verdicts.append("neutral")
        return verdicts


@pytest.mark.unit
def test_classify_plan_components_entails_expected_components() -> None:
    components = [
        PlanComponent(
            component_id="grounding",
            label="Grounding",
            hypotheses=("The plan includes grounding techniques.",),
            render="Skills: grounding.",
        ),
        PlanComponent(
            component_id="cbt",
            label="CBT",
            hypotheses=("The plan includes CBT.",),
            render="Therapy: CBT.",
        ),
    ]

    nli = _StubNLI(entail_hypotheses_containing=["grounding"])
    entailed, evidence = classify_plan_components(
        premise="anything",
        nli_model=nli,
        components=components,
    )

    assert entailed == {"grounding": True, "cbt": False}
    assert evidence == {"grounding": "The plan includes grounding techniques."}


@pytest.mark.unit
def test_render_plan_from_components_is_deterministic_and_punctuated() -> None:
    components = [
        PlanComponent(
            component_id="a",
            label="A",
            hypotheses=("h1",),
            render="Safety: do the thing",
        ),
        PlanComponent(
            component_id="b",
            label="B",
            hypotheses=("h2",),
            render="Therapy: do other thing.",
        ),
    ]

    rendered = render_plan_from_components(
        entailed_by_component_id={"a": True, "b": True},
        components=components,
    )

    assert rendered == "Safety: do the thing. Therapy: do other thing."


@pytest.mark.unit
def test_extract_recommendation_candidates_filters_and_dedups() -> None:
    text = (
        "I recommend trying breathing exercises. I recommend trying breathing exercises. "
        "We should monitor symptoms weekly. This sentence is too short."
    )

    candidates = extract_recommendation_candidates(reasoning_text=text)

    assert "I recommend trying breathing exercises." in candidates
    assert "We should monitor symptoms weekly." in candidates
    assert candidates.count("I recommend trying breathing exercises.") == 1


@pytest.mark.unit
def test_nli_filter_candidates_keeps_only_entailed() -> None:
    nli = _StubNLI(entail_hypotheses_containing=["monitor"])

    kept = nli_filter_candidates(
        premise="full premise",
        candidates=[
            "We should monitor symptoms weekly.",
            "Try a new hobby.",
        ],
        nli_model=nli,
        max_keep=3,
    )

    assert kept == ["We should monitor symptoms weekly."]
