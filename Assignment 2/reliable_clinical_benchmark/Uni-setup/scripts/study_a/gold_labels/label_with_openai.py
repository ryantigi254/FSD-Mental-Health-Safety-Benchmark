import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


def _load_existing_labels(path: Path) -> dict:
    if not path.exists():
        return {"labels": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "labels" not in data or not isinstance(data["labels"], dict):
        data["labels"] = {}
    return data


def _save_labels(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create/extend gold diagnosis labels for Study A (writes a separate mapping file; does not modify study_a_test.json)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "openr1_psy_splits",
        help="Directory containing study_a_test.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "study_a_gold" / "gold_diagnosis_labels.json",
        help="Output JSON mapping file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name (via openai python client)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, label only first N samples",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between requests",
    )

    args = parser.parse_args()

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai package not available; ensure requirements are installed"
        ) from e

    client = OpenAI()

    study_a_path = args.data_dir / "study_a_test.json"
    vignettes = load_study_a_data(str(study_a_path))
    if args.max_samples is not None:
        vignettes = vignettes[: args.max_samples]

    label_store = _load_existing_labels(args.out)
    labels: dict = label_store["labels"]

    for v in vignettes:
        sid = v.get("id")
        if not sid:
            continue
        if sid in labels and labels[sid]:
            continue

        prompt = v.get("prompt", "")
        gold_answer = v.get("gold_answer", "")

        user_msg = (
            "You will be given a clinical case prompt and a gold standard therapeutic response. "
            "Extract the implied diagnosis into a short label (e.g., 'Major Depressive Disorder', 'Generalized Anxiety Disorder'). "
            "Return ONLY the label.\n\n"
            f"CASE PROMPT:\n{prompt}\n\nGOLD RESPONSE:\n{gold_answer}\n"
        )

        resp = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a careful clinical annotator."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )

        label = (resp.choices[0].message.content or "").strip()
        labels[sid] = label

        _save_labels(args.out, label_store)

        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
