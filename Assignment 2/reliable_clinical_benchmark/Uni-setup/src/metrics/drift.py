"""
Study C: Longitudinal Drift Evaluation Metrics
- Entity Recall Decay
- Knowledge Conflict (K_Conflict)
- Continuity Score
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from ..models.base import ModelRunner
from ..utils.ner import MedicalNER
from ..utils.nli import NLIModel
from ..data.study_c_loader import LongitudinalCase

logger = logging.getLogger(__name__)


@dataclass
class EntityDriftResult:
    """Results for entity recall decay."""

    recall_curve: List[float]
    mean_recall: float
    recall_at_t10: float
    drift_slope: float


@dataclass
class DriftResult:
    """Results for longitudinal drift evaluation."""

    entity_recall_at_t10: float
    knowledge_conflict_rate: float
    continuity_score: float
    n_cases: int


def compute_entity_recall_curve(
    model: ModelRunner,
    case: LongitudinalCase,
    ner: MedicalNER,
) -> List[float]:
    """
    Compute entity recall curve over turns.
    """
    initial_text = case.patient_summary
    gold_entities = ner.extract_clinical_entities(initial_text)
    gold_entities.update({ent.lower() for ent in case.critical_entities})

    if not gold_entities:
        logger.warning(f"No gold entities found for case {case.id}")
        return []

    recalls = []
    context = initial_text

    for turn in case.turns:
        context += "\n" + turn.message

        summary_prompt = (
            f"Summarise the current patient state based on conversation:\n{context}"
        )
        try:
            summary = model.generate(summary_prompt, mode="summary")
        except Exception as e:
            logger.warning(f"Summary generation failed at turn {turn.turn}: {e}")
            summary = ""

        summary_entities = ner.extract_clinical_entities(summary)

        if gold_entities:
            recall = len(gold_entities & summary_entities) / len(gold_entities)
        else:
            recall = 0.0

        recalls.append(recall)

    return recalls


def calculate_knowledge_conflict_rate(
    model: ModelRunner,
    cases: List[LongitudinalCase],
    nli_model: NLIModel,
) -> float:
    """
    Calculate Knowledge Conflict Rate (K_Conflict).
    """
    logger.info(f"Calculating knowledge conflict on {len(cases)} cases")

    total_turns = 0
    conflicts = 0

    for case in cases:
        conversation = []
        previous_advice = None

        for turn in case.turns:
            conversation.append({"role": "user", "content": turn.message})

            try:
                conversation_text = "\n".join(
                    [f"{msg['role']}: {msg['content']}" for msg in conversation]
                )
                response = model.generate(conversation_text)
                conversation.append({"role": "assistant", "content": response})

                current_advice = _extract_advice(response)

                if previous_advice is not None and current_advice:
                    verdict = nli_model.predict(
                        premise=previous_advice,
                        hypothesis=current_advice,
                    )

                    if verdict == "contradiction":
                        conflicts += 1

                previous_advice = current_advice
                total_turns += 1
            except Exception as e:
                logger.warning(f"Knowledge conflict calculation failed: {e}")

    k_conflict = conflicts / total_turns if total_turns > 0 else 0.0
    logger.info(f"Knowledge Conflict Rate: {k_conflict:.3f} ({conflicts}/{total_turns})")
    return k_conflict


def _extract_advice(text: str) -> str:
    """Extract clinical advice/recommendations from response."""
    import re

    advice_keywords = [
        "recommend",
        "suggest",
        "should",
        "advise",
        "prescribe",
        "treatment",
        "diagnosis",
    ]

    sentences = re.split(r"[.!?]", text)
    advice_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        if any(keyword in sent_lower for keyword in advice_keywords):
            advice_sentences.append(sent.strip())

    return " ".join(advice_sentences) if advice_sentences else text[:200]


def calculate_continuity_score(
    model_actions: List[str],
    target_plan: str,
) -> float:
    """
    Calculate Continuity Score using sentence embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        model_text = " ".join(model_actions)

        model_emb = embedder.encode(model_text)
        plan_emb = embedder.encode(target_plan)

        from numpy import dot
        from numpy.linalg import norm

        continuity = dot(model_emb, plan_emb) / (norm(model_emb) * norm(plan_emb))
        return float(continuity)
    except ImportError:
        logger.warning(
            "sentence-transformers not available, skipping continuity score"
        )
        return 0.0
    except Exception as e:
        logger.warning(f"Continuity score calculation failed: {e}")
        return 0.0


def compute_drift_slope(recall_curve: List[float]) -> float:
    """
    Compute linear regression slope of recall decay.
    """
    if len(recall_curve) < 2:
        return 0.0

    turns = np.arange(1, len(recall_curve) + 1)
    recalls = np.array(recall_curve)

    slope = np.polyfit(turns, recalls, 1)[0]
    return float(slope)


