"""Study A: Faithfulness Evaluation Pipeline."""

import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterable
import logging
from datetime import datetime
import time

from ..models.base import ModelRunner
from ..metrics.faithfulness import (
    calculate_faithfulness_gap,
    calculate_step_f1,
    calculate_silent_bias_rate,
    extract_reasoning_steps,
    FaithfulnessResult,
    _is_correct_diagnosis,
)
from ..data.study_a_loader import load_study_a_data
from ..data.adversarial_loader import load_adversarial_bias_cases
from ..utils.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)


def _read_cache(cache_path: Path) -> List[Dict[str, Any]]:
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


def _write_cache_entry(cache_path: Path, entry: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False))
        f.write("\n")


def _compact_cache(cache_path: Path, make_backup: bool = True) -> None:
    """
    Compact cache to one entry per (id, mode), preferring status=ok else latest by timestamp.
    Keeps the file small (600 rows) and avoids accumulation across retries.
    """
    if not cache_path.exists():
        return
    if make_backup:
        backup = cache_path.with_suffix(
            cache_path.suffix + f".bak-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        )
        shutil.copy2(cache_path, backup)

    entries = _read_cache(cache_path)
    best: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        sid = e.get("id")
        mode = e.get("mode")
        if not sid or not mode:
            continue
        current = best.setdefault(sid, {}).get(mode)
        if current is None:
            best[sid][mode] = e
            continue
        if current.get("status") == "ok":
            if e.get("status") == "ok" and e.get("timestamp", "") > current.get("timestamp", ""):
                best[sid][mode] = e
        else:
            if e.get("status") == "ok":
                best[sid][mode] = e
            elif e.get("timestamp", "") > current.get("timestamp", ""):
                best[sid][mode] = e

    cache_path.unlink(missing_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for sid in sorted(best.keys()):
            for mode in ("cot", "direct"):
                if mode in best[sid]:
                    f.write(json.dumps(best[sid][mode], ensure_ascii=False))
                    f.write("\n")


def _existing_ok(entries: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries:
        sid = e.get("id")
        mode = e.get("mode")
        if not sid or not mode:
            continue
        if e.get("status") != "ok":
            continue
        out.setdefault(sid, {})[mode] = e
    return out


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def run_study_a(
    model: ModelRunner,
    data_dir: str = "data/openr1_psy_splits",
    adversarial_data_path: str = "data/adversarial_bias/biased_vignettes.json",
    max_samples: Optional[int] = None,
    output_dir: str = "results",
    model_name: str = "unknown",
    generate_only: bool = False,
    from_cache: Optional[str] = None,
    cache_out: Optional[str] = None,
) -> FaithfulnessResult:
    """
    Run Study A faithfulness evaluation.
    Supports generation-only, metrics-from-cache, and single-pass modes.
    """
    logger.info(f"Starting Study A evaluation for {model_name}")

    study_a_path = Path(data_dir) / "study_a_test.json"
    vignettes = load_study_a_data(str(study_a_path))

    if max_samples:
        vignettes = vignettes[:max_samples]
        logger.info(f"Limited to {max_samples} samples")

    if not vignettes:
        logger.error("No Study A data loaded. Check data paths.")
        return FaithfulnessResult(
            faithfulness_gap=0.0,
            acc_cot=0.0,
            acc_early=0.0,
            step_f1=0.0,
            silent_bias_rate=0.0,
            n_samples=0,
        )

    cache_path = Path(cache_out) if cache_out else Path(output_dir) / model_name / "study_a_generations.jsonl"
    if from_cache:
        cache_path = Path(from_cache)

    if generate_only:
        logger.info(f"Generation-only mode. Writing cache to {cache_path}")
    elif from_cache:
        logger.info(f"Metrics-from-cache mode. Loading generations from {cache_path}")
    else:
        logger.info("Single-pass mode (generate + metrics)")

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    if generate_only or not from_cache:
        _compact_cache(cache_path, make_backup=True)
        existing = {}
        if cache_path.exists():
            existing = _existing_ok(_read_cache(cache_path))
            logger.info(f"Resume enabled: found {len(existing)} cached sample(s)")

        for vignette in vignettes:
            sid = vignette.get("id")
            persona_id = vignette.get("metadata", {}).get("persona_id")
            prompt = vignette.get("prompt")
            if not sid or not prompt:
                continue
            for mode in ("cot", "direct"):
                if existing.get(sid, {}).get(mode):
                    continue
                status = "ok"
                output_text = ""
                error_message = ""
                t0 = time.perf_counter()
                try:
                    output_text = model.generate(prompt, mode=mode)
                except Exception as e:
                    status = "error"
                    error_message = str(e)
                    logger.warning(f"Generation failed for {sid} [{mode}]: {e}")
                latency_ms = int((time.perf_counter() - t0) * 1000)
                entry = {
                    "id": sid,
                    "persona_id": persona_id,
                    "mode": mode,
                    "prompt": prompt,
                    "output_text": output_text,
                    "status": status,
                    "error_message": error_message,
                    "timestamp": _now_iso(),
                    "run_id": run_id,
                    "model_name": model_name,
                    "sampling": {
                        "temperature": model.config.temperature,
                        "top_p": model.config.top_p,
                        "max_tokens": model.config.max_tokens,
                        # Placeholders for LM Studio defaults (not exposed here)
                        "top_k": None,
                        "min_p": None,
                        "repeat_penalty": None,
                        "repeat_last_n": None,
                        "dry_multiplier": None,
                        "dry_base": None,
                        "dry_allowed_length": None,
                        "dry_penalty_last_n": None,
                        "mirostat": None,
                        "mirostat_lr": None,
                        "mirostat_ent": None,
                        "logit_bias": None,
                        "n_ctx": None,
                        "n_predict": model.config.max_tokens,
                        "n_keep": None,
                        "seed": None,
                        "cache_reuse": None,
                    },
                    "meta": {
                        "prompt_tokens": None,
                        "response_tokens": None,
                        "latency_ms": latency_ms,
                    },
                }
                _write_cache_entry(cache_path, entry)

        if generate_only:
            logger.info("Generation-only complete; skipping metrics.")
            return FaithfulnessResult(
                faithfulness_gap=0.0,
                acc_cot=0.0,
                acc_early=0.0,
                step_f1=0.0,
                silent_bias_rate=0.0,
                n_samples=len(vignettes),
            )

    cache_entries = _read_cache(cache_path)
    by_id_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in cache_entries:
        sid = e.get("id")
        mode = e.get("mode")
        if sid and mode:
            by_id_mode.setdefault(sid, {})[mode] = e

    missing = [
        vid["id"]
        for vid in vignettes
        if vid["id"] not in by_id_mode
        or set(by_id_mode[vid["id"]].keys()) != {"cot", "direct"}
        or any(by_id_mode[vid["id"]][m].get("status") != "ok" for m in ("cot", "direct"))
    ]
    if missing:
        raise RuntimeError(
            f"Missing or errored generations for samples: {missing[:5]}{'...' if len(missing)>5 else ''}"
        )

    correct_cot = 0
    correct_early = 0
    for vid in vignettes:
        sid = vid["id"]
        gold = (vid.get("gold_answer") or "").lower().strip()
        cot_resp = by_id_mode[sid]["cot"]["output_text"]
        early_resp = by_id_mode[sid]["direct"]["output_text"]
        if _is_correct_diagnosis(cot_resp, gold):
            correct_cot += 1
        if _is_correct_diagnosis(early_resp, gold):
            correct_early += 1

    acc_cot = correct_cot / len(vignettes) if vignettes else 0.0
    acc_early = correct_early / len(vignettes) if vignettes else 0.0
    gap = acc_cot - acc_early

    step_f1_scores = []
    for vid in vignettes:
        sid = vid["id"]
        cot_resp = by_id_mode[sid]["cot"]["output_text"]
        model_steps = extract_reasoning_steps(cot_resp)
        gold_steps = vid.get("gold_reasoning", [])
        if gold_steps:
            f1 = calculate_step_f1(model_steps, gold_steps)
            step_f1_scores.append(f1)
    avg_step_f1 = sum(step_f1_scores) / len(step_f1_scores) if step_f1_scores else 0.0

    adversarial_cases = load_adversarial_bias_cases(adversarial_data_path)
    r_sb = (
        calculate_silent_bias_rate(model, adversarial_cases)
        if adversarial_cases and not from_cache
        else 0.0
    )

    result = FaithfulnessResult(
        faithfulness_gap=gap,
        acc_cot=acc_cot,
        acc_early=acc_early,
        step_f1=avg_step_f1,
        silent_bias_rate=r_sb,
        n_samples=len(vignettes),
    )

    output_path = Path(output_dir) / model_name / "study_a_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = {
        "model": model_name,
        "study": "A",
        "faithfulness_gap": gap,
        "acc_cot": acc_cot,
        "acc_early": acc_early,
        "step_f1": avg_step_f1,
        "silent_bias_rate": r_sb,
        "n_samples": len(vignettes),
    }

    if len(vignettes) > 10:
        gap_ci = bootstrap_confidence_interval([1.0 if gap > 0 else 0.0] * len(vignettes))
        result_dict["faithfulness_gap_ci"] = {
            "lower": gap_ci[1],
            "upper": gap_ci[2],
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Study A results saved to {output_path}")
    logger.info(
        f"Faithfulness Gap: {gap:.3f}, Step-F1: {avg_step_f1:.3f}, "
        f"Silent Bias Rate: {r_sb:.3f}"
    )

    return result

