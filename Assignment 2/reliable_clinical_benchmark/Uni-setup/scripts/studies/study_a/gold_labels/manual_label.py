import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"labels": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "labels" not in data or not isinstance(data["labels"], dict):
        data["labels"] = {}
    return data


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _truncate(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    return s.strip()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_model_consensus(processed_dir: Path) -> dict:
    consensus = {}
    if not processed_dir.exists():
        return consensus

    counts = {}

    for model_dir in processed_dir.iterdir():
        if not model_dir.is_dir():
            continue
        p = model_dir / "study_a_extracted.jsonl"
        if not p.exists():
            continue
        for row in _iter_jsonl(p):
            if row.get("is_refusal"):
                continue
            label = row.get("extracted_diagnosis")
            if not label or label in ("EXTRACTION_FAILED", "NO_OUTPUT"):
                continue
            sid = row.get("id")
            if not sid:
                continue
            key = _norm_label(label)
            if not key:
                continue
            counts.setdefault(sid, {})[key] = counts.setdefault(sid, {}).get(key, 0) + 1

    for sid, c in counts.items():
        best = sorted(c.items(), key=lambda x: (-x[1], x[0]))[:5]
        consensus[sid] = best

    return consensus


def _load_openr1_suggestions(data_dir: Path) -> dict:
    """Load diagnosis suggestions extracted from original OpenR1-Psy dataset."""
    # Try new location first
    suggestions_path = Path("data/study_a_gold/diagnosis_suggestions.json")
    if not suggestions_path.exists():
        # Fallback to old location
        suggestions_path = data_dir / "study_a_diagnosis_suggestions.json"
    if not suggestions_path.exists():
        return {}
    
    with suggestions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to simple dict: {id: diagnosis}
    result = {}
    for sid, info in data.items():
        if isinstance(info, dict) and "diagnosis" in info:
            result[sid] = {
                "diagnosis": info["diagnosis"],
                "confidence": info.get("confidence", "unknown"),
                "source": info.get("source", "unknown"),
            }
        elif isinstance(info, str):
            result[sid] = {"diagnosis": info, "confidence": "unknown", "source": "unknown"}
    
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "openr1_psy_splits",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data") / "study_a_gold" / "gold_diagnosis_labels.json",
    )
    p.add_argument(
        "--only-unlabeled",
        action="store_true",
        help="Only iterate ids with empty label",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the iteration order (after filtering)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        help="Limit number of labeled items in this session",
    )
    p.add_argument(
        "--show-consensus",
        action="store_true",
        help="Show top-5 extracted diagnosis suggestions from processed/study_a_extracted",
    )
    p.add_argument(
        "--show-openr1",
        action="store_true",
        help="Show diagnosis suggestions from original OpenR1-Psy dataset",
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "study_a_extracted",
    )
    p.add_argument(
        "--truncate",
        type=int,
        default=900,
    )

    args = p.parse_args()

    study_a_path = args.data_dir / "study_a_test.json"
    vignettes = load_study_a_data(str(study_a_path))

    label_data = _load_json(args.labels)
    labels = label_data["labels"]

    for v in vignettes:
        sid = v.get("id")
        if sid and sid not in labels:
            labels[sid] = ""

    consensus = _load_model_consensus(args.processed_dir) if args.show_consensus else {}
    openr1_suggestions = _load_openr1_suggestions(args.data_dir) if args.show_openr1 else {}

    rows = []
    for v in vignettes:
        sid = v.get("id")
        if not sid:
            continue
        gold = (labels.get(sid) or "").strip()
        if args.only_unlabeled and gold:
            continue
        rows.append(v)

    rows = rows[args.start :]

    labeled_this_session = 0
    idx = 0
    while idx < len(rows):
        v = rows[idx]
        sid = v.get("id")
        current = (labels.get(sid) or "").strip()

        print("\n" + "=" * 80)
        print(f"ID: {sid}")
        print(f"CURRENT LABEL: {current if current else '<empty>'}")

        if args.show_consensus:
            sugg = consensus.get(sid, [])
            if sugg:
                s = ", ".join([f"{k}({n})" for k, n in sugg])
                print(f"MODEL CONSENSUS: {s}")
        
        if args.show_openr1:
            openr1 = openr1_suggestions.get(sid)
            if openr1:
                conf_marker = "✓" if openr1["confidence"] == "high" else "~"
                print(f"OPENR1-PSY SUGGESTION: {conf_marker} {openr1['diagnosis']} (from {openr1['source']}, {openr1['confidence']} confidence)")

        prompt = _truncate(v.get("prompt", ""), args.truncate)
        gold_answer = _truncate(v.get("gold_answer", ""), args.truncate)

        print("\nPROMPT:\n" + prompt)
        print("\nGOLD ANSWER:\n" + gold_answer)

        user_in = input("\nEnter diagnosis label (or :skip / :q / :back / :clear): ").strip()

        if user_in == ":q":
            break
        if user_in == ":skip" or user_in == "":
            idx += 1
            continue
        if user_in == ":back":
            idx = max(0, idx - 1)
            continue
        if user_in == ":clear":
            labels[sid] = ""
            _save_json(args.labels, label_data)
            idx += 1
            continue

        labels[sid] = user_in
        _save_json(args.labels, label_data)
        labeled_this_session += 1
        idx += 1

        if args.max is not None and labeled_this_session >= args.max:
            break

    _save_json(args.labels, label_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


