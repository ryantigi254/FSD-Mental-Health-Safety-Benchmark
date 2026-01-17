"""Study C: Longitudinal Drift Evaluation Pipeline."""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import logging

from ..models.base import ModelRunner
from ..metrics.drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate,
    calculate_continuity_score,
    compute_drift_slope,
    DriftResult,
)
from ..data.study_c_loader import load_study_c_data
from ..utils.ner import MedicalNER
from ..utils.nli import NLIModel
from ..utils.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _get_max_messages_for_model(model_name: str) -> int:
    """
    Get appropriate truncation window based on model's context limit.
    
    All models in Study C support 32,768 token context length. With 10 turns
    per case (20 messages: user + assistant), we need at least 20 messages.
    Using 50 messages (~25 turns) provides safety margin for long responses.
    
    **Important**: Ensure all models are configured for 32,768 token context
    in LM Studio (Model Settings → Context Length).
    
    Args:
        model_name: Model identifier (e.g., "gpt-oss-20b", "qwen3-8b")
    
    Returns:
        Maximum number of messages to keep in conversation history (50 for 32K context)
    """
    # All models support 32K context - use generous window for 10-turn conversations
    # 50 messages = ~25 turns, well within 32K token limit even with long responses
    return 50


def _truncate_conversation_history(
    history: List[Dict[str, str]], model_name: str = "unknown", max_messages: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Truncate conversation history to prevent context length overflow.
    
    Keeps the most recent N messages (sliding window) to stay under model's
    context limit. All Study C models support 32,768 tokens, allowing for
    50 messages (~25 turns) comfortably, well beyond Study C's 10-turn requirement.
    
    **Important**: Ensure ALL models are configured for 32,768 token context
    in LM Studio (Model Settings → Context Length). This includes:
    - GPT-OSS-20B: Must be set to 32,768 (not default 4,096)
    - Qwen3-8B, QwQ-32B, DeepSeek-R1-14B, PsyLLM: 32,768 tokens
    - All local HF models: 32,768 tokens
    
    Args:
        history: List of message dicts with "role" and "content" keys
        model_name: Model identifier to determine appropriate truncation window
        max_messages: Override default (None = auto-detect based on model_name, default: 50)
    
    Returns:
        Truncated history with most recent messages preserved
    """
    if max_messages is None:
        max_messages = _get_max_messages_for_model(model_name)
    
    if len(history) <= max_messages:
        return history
    
    # Keep the most recent N messages (sliding window)
    # This preserves recent context while staying under token limits
    truncated = history[-max_messages:]
    logger.warning(
        f"Truncated conversation history from {len(history)} to {len(truncated)} messages "
        f"(model: {model_name}, limit: {max_messages}) to prevent context overflow. "
        f"Consider increasing LM Studio context length if this happens frequently."
    )
    return truncated


def _read_cache(cache_path: Path) -> List[Dict[str, Any]]:
    """Read all entries from cache JSONL file."""
    entries: List[Dict[str, Any]] = []
    if not cache_path.exists():
        return entries
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _compact_cache(cache_path: Path, make_backup: bool = True) -> None:
    """
    Compact cache to one entry per (case_id, turn_num, variant), preferring status=ok else latest by timestamp.
    Keeps the file small and avoids accumulation across retries.
    """
    if not cache_path.exists():
        return
    if make_backup:
        backup = cache_path.with_suffix(
            cache_path.suffix + f".bak-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )
        shutil.copy2(cache_path, backup)

    entries = _read_cache(cache_path)
    best: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for e in entries:
        case_id = e.get("case_id")
        turn_num = e.get("turn_num")
        variant = e.get("variant")
        if not case_id or not isinstance(turn_num, int) or not variant:
            continue
        case_key = str(case_id)
        turn_key = int(turn_num)
        variant_key = str(variant)
        
        current = best.setdefault(case_key, {}).setdefault(turn_key, {}).get(variant_key)
        if current is None:
            best[case_key][turn_key][variant_key] = e
            continue
        if current.get("status") == "ok":
            if e.get("status") == "ok" and e.get("timestamp", "") > current.get("timestamp", ""):
                best[case_key][turn_key][variant_key] = e
        else:
            if e.get("status") == "ok":
                best[case_key][turn_key][variant_key] = e
            elif e.get("timestamp", "") > current.get("timestamp", ""):
                best[case_key][turn_key][variant_key] = e

    cache_path.unlink(missing_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for case_id in sorted(best.keys()):
            for turn_num in sorted(best[case_id].keys()):
                for variant in sorted(best[case_id][turn_num].keys()):
                    f.write(json.dumps(best[case_id][turn_num][variant], ensure_ascii=False))
                    f.write("\n")


def _existing_ok(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    """
    Build lookup of existing OK entries keyed by (case_id, turn_num, variant).
    Returns: Dict[case_id][turn_num][variant] = entry
    """
    out: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
    for e in entries:
        if e.get("status") != "ok":
            continue
        case_id = e.get("case_id")
        turn_num = e.get("turn_num")
        variant = e.get("variant")
        if not case_id or not isinstance(turn_num, int) or not variant:
            continue
        out.setdefault(str(case_id), {}).setdefault(int(turn_num), {})[str(variant)] = e
    return out


def _write_cache_entry(cache_path: Path, entry: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")


def run_study_c(
    model: ModelRunner,
    data_dir: str = "data/openr1_psy_splits",
    max_cases: Optional[int] = None,
    output_dir: str = "results",
    model_name: str = "unknown",
    use_nli: bool = True,
    generate_only: bool = False,
    cache_out: Optional[str] = None,
) -> DriftResult:
    """
    Run Study C longitudinal drift evaluation.

    Args:
        model: ModelRunner instance
        data_dir: Directory containing study_c_test.json
        max_cases: Maximum number of cases to evaluate (None = all)
        output_dir: Directory to save results
        model_name: Name of the model being evaluated
        use_nli: Whether to compute knowledge conflict (requires NLI model)
        generate_only: If True, write generations JSONL only (no metrics).
        cache_out: Path to write cached generations JSONL when using generate_only.

    Returns:
        DriftResult with all metrics
    """
    logger.info(f"Starting Study C evaluation for {model_name}")

    # Load data
    study_c_path = Path(data_dir) / "study_c_test.json"
    cases = load_study_c_data(str(study_c_path))

    if max_cases:
        cases = cases[:max_cases]
        logger.info(f"Limited to {max_cases} cases")

    if not cases:
        logger.error("No Study C data loaded. Check data paths.")
        return DriftResult(
            entity_recall_at_t10=0.0,
            knowledge_conflict_rate=0.0,
            continuity_score=None,
            n_cases=0,
        )

    cache_path = Path(cache_out) if cache_out else Path(output_dir) / model_name / "study_c_generations.jsonl"
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    if generate_only:
        logger.info(f"Generation-only mode. Writing Study C cache to {cache_path}")
        
        _compact_cache(cache_path, make_backup=True)
        existing: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = {}
        if cache_path.exists():
            entries = _read_cache(cache_path)
            existing = _existing_ok(entries)
            total_cached = sum(
                len(variants) for case_dict in existing.values() for variants in case_dict.values()
            )
            logger.info(f"Resume enabled: found {total_cached} cached (case, turn, variant) entry/entries")

        for case in cases:
            persona_id = (case.metadata or {}).get("persona_id")
            context_for_summary = case.patient_summary
            conversation_history = []

            for turn in case.turns:
                # Summary-style prompt used by compute_entity_recall_curve()
                context_for_summary += "\n" + turn.message
                summary_prompt = (
                    f"Summarise the current patient state based on conversation:\n{context_for_summary}"
                )

                # Check if already generated
                if existing.get(case.id, {}).get(turn.turn, {}).get("summary"):
                    logger.debug(f"Skipping case {case.id} turn {turn.turn} variant summary (already cached)")
                else:
                    status = "ok"
                    summary_text = ""
                    error_message = ""
                    t0 = time.perf_counter()
                    max_retries = 3
                    retry_delay = 2.0  # seconds
                    for attempt in range(max_retries):
                        try:
                            summary_text = model.generate(summary_prompt, mode="summary")
                            break  # Success, exit retry loop
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Summary generation failed for case {case.id} turn {turn.turn} "
                                    f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                status = "error"
                                error_message = str(e)
                                logger.error(
                                    f"Summary generation failed for case {case.id} turn {turn.turn} "
                                    f"after {max_retries} attempts: {e}"
                                )
                    latency_ms = int((time.perf_counter() - t0) * 1000)

                    _write_cache_entry(
                        cache_path,
                        {
                            "case_id": case.id,
                            "persona_id": persona_id,
                            "turn_num": turn.turn,
                            "variant": "summary",
                            "prompt": summary_prompt,
                            "response_text": summary_text,
                            "status": status,
                            "error_message": error_message,
                            "timestamp": _now_iso(),
                            "run_id": run_id,
                            "model_name": model_name,
                            "meta": {"latency_ms": latency_ms},
                        },
                    )

                # Dialogue-style prompt used by calculate_knowledge_conflict_rate()
                conversation_history.append({"role": "user", "content": turn.message})
                
                # Full conversation history maintained (all models support 32K context, 10 turns = 20 messages)

                # Check if already generated
                if existing.get(case.id, {}).get(turn.turn, {}).get("dialogue"):
                    logger.debug(f"Skipping case {case.id} turn {turn.turn} variant dialogue (already cached)")
                    # Still need to append to conversation_history for next turn
                    cached_dialogue = existing[case.id][turn.turn]["dialogue"]
                    response_text = cached_dialogue.get("response_text", "")
                    if response_text:
                        conversation_history.append({"role": "assistant", "content": response_text})
                        # Full conversation history maintained for subsequent turns
                else:
                    status = "ok"
                    response_text = ""
                    error_message = ""
                    t0 = time.perf_counter()
                    max_retries = 3
                    retry_delay = 2.0  # seconds
                    for attempt in range(max_retries):
                        try:
                            # Use model.chat() with structured conversation history (like Study B)
                            # This properly handles multi-turn conversations with LM Studio's chat completion API
                            # Full conversation history maintained (all models support 32K context)
                            response_text = model.chat(conversation_history, mode="default")
                            conversation_history.append({"role": "assistant", "content": response_text})
                            break  # Success, exit retry loop
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(
                                    f"Dialogue generation failed for case {case.id} turn {turn.turn} "
                                    f"(attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay}s..."
                                )
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                status = "error"
                                error_message = str(e)
                                logger.error(
                                    f"Dialogue generation failed for case {case.id} turn {turn.turn} "
                                    f"after {max_retries} attempts: {e}"
                                )
                    latency_ms = int((time.perf_counter() - t0) * 1000)

                    # Generate conversation_text for logging/debugging (after response is added to history)
                    conversation_text = "\n".join(
                        [f"{msg['role']}: {msg['content']}" for msg in conversation_history[:-1]]  # Exclude last assistant response
                    )

                    _write_cache_entry(
                        cache_path,
                        {
                            "case_id": case.id,
                            "persona_id": persona_id,
                            "turn_num": turn.turn,
                            "variant": "dialogue",
                            "conversation_text": conversation_text,
                            "response_text": response_text,
                            "status": status,
                            "error_message": error_message,
                            "timestamp": _now_iso(),
                            "run_id": run_id,
                            "model_name": model_name,
                            "meta": {"latency_ms": latency_ms},
                        },
                    )

        logger.info("Study C generation-only complete; skipping metrics.")
        return DriftResult(
            entity_recall_at_t10=0.0,
            knowledge_conflict_rate=0.0,
            continuity_score=None,
            n_cases=len(cases),
        )

    # Initialise NER
    try:
        ner = MedicalNER()
    except Exception as e:
        logger.error(f"Failed to load NER model: {e}")
        return DriftResult(
            entity_recall_at_t10=0.0,
            knowledge_conflict_rate=0.0,
            continuity_score=None,
            n_cases=0,
        )

    # Compute entity recall curves
    all_recalls_at_t10 = []
    all_recall_curves = []

    for case in cases:
        try:
            recall_curve = compute_entity_recall_curve(model, case, ner)
            if recall_curve:
                all_recall_curves.append(recall_curve)
                # Get recall at turn 10 (or last turn if fewer than 10)
                recall_at_t10 = (
                    recall_curve[9] if len(recall_curve) > 9 else recall_curve[-1]
                )
                all_recalls_at_t10.append(recall_at_t10)
        except Exception as e:
            logger.warning(f"Entity recall calculation failed for case {case.id}: {e}")

    mean_recall_at_t10 = (
        sum(all_recalls_at_t10) / len(all_recalls_at_t10)
        if all_recalls_at_t10
        else 0.0
    )

    # Calculate knowledge conflict rate
    k_conflict = 0.0
    if use_nli:
        try:
            nli_model = NLIModel()
            k_conflict = calculate_knowledge_conflict_rate(model, cases, nli_model)
        except Exception as e:
            logger.warning(f"NLI model not available, skipping knowledge conflict: {e}")

    continuity_score = None

    result = DriftResult(
        entity_recall_at_t10=mean_recall_at_t10,
        knowledge_conflict_rate=k_conflict,
        continuity_score=continuity_score,
        n_cases=len(cases),
    )

    # Save results
    output_path = Path(output_dir) / model_name / "study_c_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "model": model_name,
        "study": "C",
        "entity_recall_at_t10": mean_recall_at_t10,
        "knowledge_conflict_rate": k_conflict,
        "n_cases": len(cases),
    }

    if continuity_score is not None:
        result_dict["continuity_score"] = continuity_score

    # Add bootstrap CIs if we have enough cases
    if len(all_recalls_at_t10) > 10:
        recall_ci = bootstrap_confidence_interval(all_recalls_at_t10)
        result_dict["entity_recall_ci"] = {
            "lower": recall_ci[1],
            "upper": recall_ci[2],
        }

    # Save average recall curve
    if all_recall_curves:
        # Average across all cases
        max_turns = max(len(curve) for curve in all_recall_curves)
        avg_curve = []
        for turn_idx in range(max_turns):
            turn_recalls = [
                curve[turn_idx] for curve in all_recall_curves if len(curve) > turn_idx
            ]
            if turn_recalls:
                avg_curve.append(sum(turn_recalls) / len(turn_recalls))
        result_dict["average_recall_curve"] = avg_curve

    with open(output_path, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Study C results saved to {output_path}")
    logger.info(
        f"Entity Recall (T=10): {mean_recall_at_t10:.3f}, "
        f"Knowledge Conflict: {k_conflict:.3f}"
    )

    return result

