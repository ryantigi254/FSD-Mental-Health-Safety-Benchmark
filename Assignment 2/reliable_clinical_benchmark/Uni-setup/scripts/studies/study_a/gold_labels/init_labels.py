import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from reliable_clinical_benchmark.data.study_a_loader import load_study_a_data


def _load_mapping(path: Path) -> dict:
    if not path.exists():
        return {"labels": {}}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "labels" not in data or not isinstance(data["labels"], dict):
        data["labels"] = {}
    return data


def _save_mapping(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data") / "openr1_psy_splits",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "study_a_gold" / "gold_diagnosis_labels.json",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace labels mapping entirely (default: merge/extend)",
    )

    args = p.parse_args()

    study_a_path = args.data_dir / "study_a_test.json"
    vignettes = load_study_a_data(str(study_a_path))

    if args.overwrite:
        mapping = {"labels": {}}
    else:
        mapping = _load_mapping(args.out)

    labels = mapping["labels"]

    for v in vignettes:
        sid = v.get("id")
        if not sid:
            continue
        labels.setdefault(sid, "")

    _save_mapping(args.out, mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


