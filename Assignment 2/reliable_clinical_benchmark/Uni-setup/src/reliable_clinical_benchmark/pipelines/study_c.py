"""Study C: Longitudinal Drift Evaluation Pipeline."""

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from rapidfuzz import fuzz, utils
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    logger.warning(
        "rapidfuzz not available. Install with: pip install rapidfuzz. "
        "Falling back to basic repetition detection."
    )

from ..models.base import ModelRunner
from ..metrics.drift import (
    compute_entity_recall_curve,
    calculate_knowledge_conflict_rate_from_responses,
    calculate_continuity_score,
    compute_drift_slope,
    DriftResult,
)
from ..data.study_c_loader import load_study_c_data
from ..utils.ner import MedicalNER
from ..utils.nli import NLIModel
from ..utils.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)


def _remove_repetition(text: str, max_repetition_ratio: float = 0.3, min_repeat_length: int = 50) -> str:
    """
    Remove excessive repetition from model output to prevent memory bloat in conversation history.
    
    Uses rapidfuzz library for robust repetition detection with fuzzy matching capabilities.
    Falls back to basic detection if rapidfuzz is not available.
    
    Uses intelligent sequence matching to detect truly repeated content, not just similar text.
    Only removes content that is REPEATED multiple times, preserving important context.
    
    NOTE ON FAIRNESS: This cleaning is applied ONLY to conversation history (for memory efficiency),
    NOT to saved responses (which remain raw for evaluation). This may affect longitudinal drift
    measurements - see report methodology section for limitations discussion.
    
    Detection Strategy (with rapidfuzz):
    1. Uses rapidfuzz for robust fuzzy matching of repeated sequences
    2. Detects consecutive identical/near-identical blocks (similarity >95%)
    3. Finds repeated sentence sequences using sliding window
    4. Only removes if repetition is substantial (>30% of text) and appears 3+ times
    5. Preserves first occurrence and all unique content
    
    Args:
        text: Raw model output
        max_repetition_ratio: Maximum ratio of repeated content (default: 0.3 = 30%)
        min_repeat_length: Minimum length of repeated sequence to consider (default: 50 chars)
    
    Returns:
        Cleaned text with repetition removed, or original text if no excessive repetition detected
    """
    if not text or len(text) < 200:  # Need substantial text to detect repetition
        return text
    
    original_text = text
    
    # Use rapidfuzz if available for more robust detection
    if RAPIDFUZZ_AVAILABLE:
        return _remove_repetition_rapidfuzz(text, max_repetition_ratio, min_repeat_length)
    else:
        # Fallback to basic detection
        return _remove_repetition_basic(text, max_repetition_ratio, min_repeat_length)


def _remove_repetition_rapidfuzz(text: str, max_repetition_ratio: float, min_repeat_length: int) -> str:
    """
    Advanced repetition detection using rapidfuzz library.
    
    Uses rapidfuzz's fuzzy matching to detect repeated content with high accuracy.
    Can detect both exact duplicates and near-duplicates (similarity >95%).
    """
    original_text = text
    
    # Strategy 1: Detect consecutive identical/near-identical long lines using rapidfuzz
    # This catches cases like "Final Final Final Answer:" repeated many times
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 5:
        # Use rapidfuzz to find consecutive similar lines (fuzzy matching)
        # Check last 20 lines for repetition patterns
        recent_lines = lines[-20:] if len(lines) > 20 else lines
        
        # Find consecutive similar lines using rapidfuzz ratio
        consecutive_repeats = []
        if len(recent_lines) >= 3:
            # Start from the end and work backwards
            i = len(recent_lines) - 1
            base_line = recent_lines[i]
            
            if len(base_line) > min_repeat_length:
                # Count consecutive lines with >95% similarity
                while i >= 0:
                    similarity = fuzz.ratio(base_line, recent_lines[i], score_cutoff=95)
                    if similarity >= 95:
                        consecutive_repeats.append(i)
                        i -= 1
                    else:
                        break
                
                # If we found 3+ consecutive similar lines, truncate before them
                if len(consecutive_repeats) >= 3:
                    # Find where the repetition starts in original text
                    first_repeat_idx = len(lines) - len(recent_lines) + min(consecutive_repeats)
                    kept_lines = lines[:first_repeat_idx]
                    text = '\n'.join(kept_lines)
                    if text and not text.endswith('\n'):
                        text += '\n'
                    text += "[Repetitive content removed]"
                    logger.info(
                        f"Removed {len(consecutive_repeats)} consecutive similar lines "
                        f"(similarity >95%, length: {len(base_line)} chars) from response"
                    )
                    return text
    
    # Strategy 2: Detect repeated sentence sequences using rapidfuzz fuzzy matching
    # Split into sentences (using multiple delimiters)
    sentences = re.split(r'([.!?]+\s+)', text)
    # Recombine sentences with their punctuation
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2) if sentences[i].strip()]
    
    if len(sentences) < 5:
        return original_text  # Not enough sentences to detect repetition
    
    # Find repeated sequences using rapidfuzz for fuzzy matching
    # Check for sequences of 2-5 sentences that repeat
    max_window = min(5, len(sentences) // 3)  # Don't check windows larger than 1/3 of text
    
    for window_size in range(max_window, 1, -1):  # Start with larger windows
        for start_idx in range(len(sentences) - window_size * 2):
            # Extract a sequence and normalize
            sequence_text = ' '.join(s.strip() for s in sentences[start_idx:start_idx + window_size])
            sequence_normalized = utils.default_process(sequence_text)  # rapidfuzz normalization
            
            # Skip if sequence contains very short sentences (likely not meaningful repetition)
            if len(sequence_normalized) < min_repeat_length:
                continue
            
            # Count how many times this sequence appears (with fuzzy matching >95% similarity)
            matches = []
            for check_idx in range(start_idx + window_size, len(sentences) - window_size + 1):
                check_sequence_text = ' '.join(s.strip() for s in sentences[check_idx:check_idx + window_size])
                check_sequence_normalized = utils.default_process(check_sequence_text)
                
                # Use rapidfuzz ratio for fuzzy matching
                similarity = fuzz.ratio(sequence_normalized, check_sequence_normalized, score_cutoff=95)
                if similarity >= 95:
                    matches.append(check_idx)
            
            # If sequence appears 3+ times, it's excessive repetition
            if len(matches) >= 2:  # Original + 2 repeats = 3 total occurrences
                # Keep first occurrence, remove subsequent ones
                # But only if repetition is substantial (>30% of remaining text)
                total_sentences_after_first = len(sentences) - start_idx - window_size
                if total_sentences_after_first > 0:
                    repetition_ratio = (len(matches) * window_size) / total_sentences_after_first
                    
                    if repetition_ratio > max_repetition_ratio:
                        # Keep everything up to and including first occurrence
                        kept_sentences = sentences[:start_idx + window_size]
                        text = ''.join(kept_sentences).strip()
                        text += "\n\n[Repetitive content removed]"
                        logger.info(
                            f"Removed repeated sequence of {window_size} sentences "
                            f"(appeared {len(matches) + 1} times, similarity >95%, ratio: {repetition_ratio:.2f})"
                        )
                        return text
    
    # Strategy 3: Detect very long repeated substrings using rapidfuzz
    # Use rapidfuzz's partial_ratio for substring matching
    text_normalized = utils.default_process(text)
    n = len(text_normalized)
    
    # Check for repeated substrings of substantial length
    # Use sliding window to find repeated chunks
    for substr_len in range(min(200, n // 4), min_repeat_length - 1, -20):  # Check in steps
        for i in range(n - substr_len * 2):
            substr = text_normalized[i:i + substr_len]
            
            # Count occurrences using rapidfuzz partial_ratio (for substring matching)
            matches = []
            search_start = i + substr_len
            while search_start < n - substr_len:
                # Use partial_ratio to find similar substrings
                similarity = fuzz.partial_ratio(substr, text_normalized[search_start:], score_cutoff=95)
                if similarity >= 95:
                    # Find exact position
                    check_substr = text_normalized[search_start:search_start + substr_len]
                    if fuzz.ratio(substr, check_substr, score_cutoff=95) >= 95:
                        matches.append(search_start)
                        search_start += substr_len  # Skip past this match
                    else:
                        search_start += 1
                else:
                    search_start += 1
            
            # If substring appears 3+ times consecutively, it's excessive repetition
            if len(matches) >= 2:  # Original + 2 repeats = 3 total occurrences
                # Check if matches are consecutive (within small margin)
                if len(matches) >= 2:
                    gaps = [matches[i+1] - matches[i] for i in range(len(matches)-1)]
                    if all(gap <= substr_len + 50 for gap in gaps):  # Allow small gaps
                        # This is consecutive repetition - truncate before first repeat
                        truncate_pos = matches[0]
                        # Convert normalized position back to original text position (approximate)
                        original_pos = int(truncate_pos * len(text) / len(text_normalized))
                        text = text[:original_pos].rstrip()
                        text += "\n\n[Repetitive content removed]"
                        logger.info(
                            f"Removed repeated substring of length {substr_len} chars "
                            f"(appeared {len(matches) + 1} times consecutively, similarity >95%)"
                        )
                        return text
    
    # No excessive repetition detected - return original
    return original_text


def _remove_repetition_basic(text: str, max_repetition_ratio: float, min_repeat_length: int) -> str:
    """
    Basic repetition detection fallback (used when rapidfuzz is not available).
    
    Uses simple exact matching for repetition detection.
    """
    original_text = text
    
    # Strategy 1: Detect consecutive identical long lines
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 5:
        i = len(lines) - 1
        consecutive_repeats = []
        current_line = lines[i] if i >= 0 else ""
        
        while i >= 0 and lines[i] == current_line and len(current_line) > min_repeat_length:
            consecutive_repeats.append(i)
            i -= 1
        
        if len(consecutive_repeats) >= 3:
            first_repeat_idx = min(consecutive_repeats)
            kept_lines = lines[:first_repeat_idx]
            text = '\n'.join(kept_lines)
            if text and not text.endswith('\n'):
                text += '\n'
            text += "[Repetitive content removed]"
            logger.info(
                f"Removed {len(consecutive_repeats)} consecutive identical lines "
                f"(length: {len(current_line)} chars) from response"
            )
            return text
    
    # Strategy 2: Detect repeated sentence sequences (exact matching)
    sentences = re.split(r'([.!?]+\s+)', text)
    sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                 for i in range(0, len(sentences), 2) if sentences[i].strip()]
    
    if len(sentences) < 5:
        return original_text
    
    max_window = min(5, len(sentences) // 3)
    for window_size in range(max_window, 1, -1):
        for start_idx in range(len(sentences) - window_size * 2):
            sequence = tuple(s.strip().lower() for s in sentences[start_idx:start_idx + window_size])
            if any(len(s) < 30 for s in sequence):
                continue
            
            matches = []
            for check_idx in range(start_idx + window_size, len(sentences) - window_size + 1):
                check_sequence = tuple(s.strip().lower() for s in sentences[check_idx:check_idx + window_size])
                if sequence == check_sequence:
                    matches.append(check_idx)
            
            if len(matches) >= 2:
                total_sentences_after_first = len(sentences) - start_idx - window_size
                if total_sentences_after_first > 0:
                    repetition_ratio = (len(matches) * window_size) / total_sentences_after_first
                    if repetition_ratio > max_repetition_ratio:
                        kept_sentences = sentences[:start_idx + window_size]
                        text = ''.join(kept_sentences).strip()
                        text += "\n\n[Repetitive content removed]"
                        logger.info(
                            f"Removed repeated sequence of {window_size} sentences "
                            f"(appeared {len(matches) + 1} times, ratio: {repetition_ratio:.2f})"
                        )
                        return text
    
    return original_text


def _load_study_c_target_plans(study_c_split_path: Path) -> Dict[str, str]:
    """Load Study C gold target plans if present.

    Expected location (mirrors Study A gold approach):
    data/study_c_gold/target_plans.json
    (sibling of data/openr1_psy_splits/)
    """
    candidate = (
        study_c_split_path.parent.parent / "study_c_gold" / "target_plans.json"
    )
    if not candidate.exists():
        return {}

    try:
        with candidate.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load Study C target plans from {candidate}: {e}")
        return {}

    raw_plans = payload.get("plans", payload)
    if not isinstance(raw_plans, dict):
        return {}

    out: Dict[str, str] = {}
    for case_id, v in raw_plans.items():
        if isinstance(v, str):
            out[str(case_id)] = v
        elif isinstance(v, dict):
            plan = v.get("plan")
            if plan is not None:
                out[str(case_id)] = str(plan)
    return out


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

    target_plans_by_case_id = _load_study_c_target_plans(study_c_path)

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
                        # Clean repetitive text before adding to conversation history
                        cleaned_response = _remove_repetition(response_text)
                        conversation_history.append({"role": "assistant", "content": cleaned_response})
                        # Full conversation history maintained for subsequent turns
                else:
                    status = "ok"
                    response_text = ""
                    error_message = ""
                    t0 = time.perf_counter()
                    max_retries = 3
                    retry_delay = 2.0  # seconds
                    original_max_tokens = model.config.max_tokens
                    
                    for attempt in range(max_retries):
                        try:
                            # Clear GPU cache before each attempt (especially important after OOM errors)
                            if TORCH_AVAILABLE and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                # Reset peak memory stats to help with fragmentation
                                torch.cuda.reset_peak_memory_stats(0)
                            
                            # Reduce max_new_tokens progressively on retries to handle memory pressure
                            if attempt > 0:
                                # Halve max_tokens on each retry (down to minimum of 256)
                                current_max = model.config.max_tokens
                                reduced_tokens = max(256, current_max // (2 ** attempt))
                                if reduced_tokens < current_max:
                                    logger.info(
                                        f"Reducing max_tokens from {current_max} to {reduced_tokens} "
                                        f"for retry attempt {attempt + 1}/{max_retries}"
                                    )
                                    model.config.max_tokens = reduced_tokens
                            
                            # Use model.chat() with structured conversation history (like Study B)
                            # This properly handles multi-turn conversations with LM Studio's chat completion API
                            # Full conversation history maintained (all models support 32K context)
                            response_text = model.chat(conversation_history, mode="default")
                            
                            # Clean repetitive text before adding to conversation history
                            # This prevents memory bloat in future turns while keeping raw response in saved file
                            cleaned_response = _remove_repetition(response_text)
                            if cleaned_response != response_text:
                                original_len = len(response_text)
                                cleaned_len = len(cleaned_response)
                                logger.info(
                                    f"Cleaned {original_len - cleaned_len} characters of repetition "
                                    f"from turn {turn.turn} response before adding to conversation history"
                                )
                            
                            # Add cleaned response to conversation history (saves memory)
                            # But save raw response to file (for analysis)
                            conversation_history.append({"role": "assistant", "content": cleaned_response})
                            
                            # Restore original max_tokens after success
                            model.config.max_tokens = original_max_tokens
                            break  # Success, exit retry loop
                        except Exception as e:
                            # Restore original max_tokens before next retry
                            model.config.max_tokens = original_max_tokens
                            
                            # Check if it's a CUDA OOM error
                            error_str = str(e).lower()
                            is_cuda_oom = "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str)
                            
                            if attempt < max_retries - 1:
                                # Clear GPU cache more aggressively on CUDA OOM
                                if is_cuda_oom and TORCH_AVAILABLE and torch.cuda.is_available():
                                    logger.info("Clearing GPU cache aggressively after CUDA OOM error...")
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                    # Reset peak memory stats
                                    torch.cuda.reset_peak_memory_stats(0)
                                    # Force garbage collection
                                    import gc
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                
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

    # Collect model actions (dialogue responses) per case only if needed.
    has_any_target_plan = any(
        bool(target_plans_by_case_id.get(case.id, "")) for case in cases
    )
    need_dialogue_metrics = bool(use_nli) or has_any_target_plan

    responses_by_case_id: Dict[str, List[str]] = {}
    if need_dialogue_metrics:
        for case in cases:
            conversation_history: List[Dict[str, str]] = []
            responses: List[str] = []
            for turn in case.turns:
                conversation_history.append({"role": "user", "content": turn.message})
                try:
                    resp = model.chat(conversation_history, mode="default")
                except Exception as e:
                    logger.warning(
                        f"Dialogue generation failed for continuity/K_Conflict case {case.id} turn {turn.turn}: {e}"
                    )
                    resp = ""
                responses.append(resp)  # Save raw response for metrics
                # Clean repetitive text before adding to conversation history
                cleaned_resp = _remove_repetition(resp)
                conversation_history.append({"role": "assistant", "content": cleaned_resp})
            responses_by_case_id[case.id] = responses

    # Calculate knowledge conflict rate
    k_conflict = 0.0
    if use_nli and responses_by_case_id:
        try:
            nli_model = NLIModel()
            k_conflict = calculate_knowledge_conflict_rate_from_responses(
                responses_by_case_id, nli_model
            )
        except Exception as e:
            logger.warning(f"NLI model not available, skipping knowledge conflict: {e}")

    continuity_score: Optional[float] = None
    continuity_scores: List[float] = []
    for case in cases:
        plan = target_plans_by_case_id.get(case.id, "")
        if not plan:
            continue
        score = calculate_continuity_score(
            responses_by_case_id.get(case.id, []), plan
        )
        if score is not None:
            continuity_scores.append(score)

    if continuity_scores:
        continuity_score = sum(continuity_scores) / len(continuity_scores)

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

