from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Sequence, Tuple


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class PlanComponent:
    component_id: str
    label: str
    hypotheses: Sequence[str]
    render: str


DEFAULT_PLAN_COMPONENTS: List[PlanComponent] = [
    PlanComponent(
        component_id="risk_safety",
        label="Safety",
        hypotheses=(
            "The plan includes safety planning or suicide risk assessment.",
            "The plan includes a risk assessment for self-harm or suicide.",
        ),
        render="Safety: assess risk and develop a safety plan if indicated.",
    ),
    PlanComponent(
        component_id="cbt_cognitive_restructuring",
        label="CBT (cognitive restructuring)",
        hypotheses=(
            "The plan includes CBT-style cognitive restructuring (thought challenging).",
            "The plan includes identifying and challenging unhelpful thoughts.",
        ),
        render="Therapy: CBT-style cognitive restructuring (identify and challenge unhelpful thoughts).",
    ),
    PlanComponent(
        component_id="behavioural_activation",
        label="Behavioural activation",
        hypotheses=(
            "The plan includes behavioural activation (increasing meaningful activities).",
        ),
        render="Therapy: behavioural activation (schedule small, achievable activities).",
    ),
    PlanComponent(
        component_id="grounding_distress_tolerance",
        label="Grounding / distress tolerance",
        hypotheses=(
            "The plan includes grounding techniques or breathing exercises.",
            "The plan includes distress tolerance skills.",
        ),
        render="Skills: grounding/breathing to reduce acute distress.",
    ),
    PlanComponent(
        component_id="exposure",
        label="Exposure",
        hypotheses=(
            "The plan includes exposure-based work for anxiety or avoidance.",
        ),
        render="Therapy: graded exposure for avoidance triggers.",
    ),
    PlanComponent(
        component_id="medication_review_gp",
        label="Medication review",
        hypotheses=(
            "The plan recommends medication review or optimisation with a GP or psychiatrist.",
            "The plan includes reviewing SSRI or SNRI medication with a clinician.",
        ),
        render="Medication: review current medication with prescriber if symptoms persist.",
    ),
    PlanComponent(
        component_id="referral_signposting",
        label="Referral",
        hypotheses=(
            "The plan includes referral or signposting to specialist services.",
        ),
        render="Referral: signpost or refer to appropriate services if needed.",
    ),
    PlanComponent(
        component_id="homework_between_session",
        label="Between-session practice",
        hypotheses=(
            "The plan includes homework or between-session practice.",
            "The plan includes practising skills between sessions.",
        ),
        render="Between-session: agree one small homework task to practise.",
    ),
]


def classify_plan_components(
    *,
    premise: str,
    nli_model: object,
    components: Sequence[PlanComponent] = DEFAULT_PLAN_COMPONENTS,
) -> Tuple[Dict[str, bool], Dict[str, str]]:

    premise_text = str(premise or "").strip()
    if not premise_text:
        return {}, {}

    pairs: List[Tuple[str, str]] = []
    pair_meta: List[Tuple[str, str]] = []

    for component in components:
        for hypothesis in component.hypotheses:
            h = str(hypothesis or "").strip()
            if not h:
                continue
            pairs.append((premise_text, h))
            pair_meta.append((component.component_id, h))

    if not pairs:
        return {}, {}

    batch_predict = getattr(nli_model, "batch_predict", None)
    if batch_predict is None or not callable(batch_predict):
        raise TypeError("nli_model must provide a callable batch_predict(premise_hypothesis_pairs)")

    verdicts = batch_predict(pairs)

    entailed: Dict[str, bool] = {}
    evidence: Dict[str, str] = {}

    for (component_id, hypothesis), verdict in zip(pair_meta, verdicts):
        if entailed.get(component_id):
            continue
        if str(verdict).lower().strip() == "entailment":
            entailed[component_id] = True
            evidence[component_id] = hypothesis

    # Ensure keys exist for all components (explicit False)
    for component in components:
        entailed.setdefault(component.component_id, False)

    return entailed, evidence


def render_plan_from_components(
    *,
    entailed_by_component_id: Dict[str, bool],
    components: Sequence[PlanComponent] = DEFAULT_PLAN_COMPONENTS,
) -> str:

    parts: List[str] = []
    for component in components:
        if not entailed_by_component_id.get(component.component_id, False):
            continue
        text = str(component.render or "").strip()
        if not text:
            continue
        if not text.endswith("."):
            text += "."
        parts.append(text)

    return " ".join(parts).strip()


_ACTION_KEYWORDS = re.compile(
    r"\b(?:"
    r"recommend|suggest|consider|advise|propose|encourage|urge|direct|instruct|mandate|"
    r"plan|aim|goal|target|prioriti[sz]e|must|should|ought\s*to|"
    r"prescribe|dispense|administer|start|initiate|commence|introduce|"
    r"take|use|consume|inject|apply|dose|"
    r"increase|decrease|reduce|lower|raise|adjust|optimi[sz]e|modify|alter|"
    r"switch|change|rotate|substitute|replace|"
    r"titrate|taper|wean|continue|"
    r"stop|discontinue|cease|withdraw|halt|suspend|"
    r"avoid|refrain|abstain|restrict|limit|prevent|prohibit|forbid|warn|caution|"
    r"assess|evaluate|examine|screen|investigate|explore|review|"
    r"monitor|observe|track|watch|check|measure|record|log|"
    r"order|request|schedule|arrange|"
    r"refer|consult|admit|discharge|transfer|signpost|connect|link|contact|call|seek|"
    r"follow\s*up|"
    r"work\s*on|focus\s*on|practi[sc]e|engage|attend|participate|implement|utili[sz]e|employ|try"
    r")\b",
    flags=re.IGNORECASE,
)


def extract_recommendation_candidates(
    *,
    reasoning_text: str,
    min_chars: int = 25,
    max_chars: int = 300,
) -> List[str]:

    text = str(reasoning_text or "").strip()
    if not text:
        return []

    raw_sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]

    seen = set()
    out: List[str] = []
    for sent in raw_sentences:
        if len(sent) < min_chars or len(sent) > max_chars:
            continue
        if not _ACTION_KEYWORDS.search(sent):
            continue
        key = re.sub(r"\s+", " ", sent.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(sent)

    return out


def nli_filter_candidates(
    *,
    premise: str,
    candidates: Sequence[str],
    nli_model: object,
    max_keep: int = 3,
) -> List[str]:

    premise_text = str(premise or "").strip()
    if not premise_text:
        return []

    cleaned_candidates = [str(c or "").strip() for c in candidates if str(c or "").strip()]
    if not cleaned_candidates:
        return []

    pairs = [(premise_text, c) for c in cleaned_candidates]

    batch_predict = getattr(nli_model, "batch_predict", None)
    if batch_predict is None or not callable(batch_predict):
        raise TypeError("nli_model must provide a callable batch_predict(premise_hypothesis_pairs)")

    verdicts = batch_predict(pairs)

    kept: List[str] = []
    for cand, verdict in zip(cleaned_candidates, verdicts):
        if str(verdict).lower().strip() == "entailment":
            kept.append(cand)
            if len(kept) >= max_keep:
                break

    return kept
