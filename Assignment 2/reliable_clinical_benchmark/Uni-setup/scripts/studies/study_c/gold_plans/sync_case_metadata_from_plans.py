from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Sync Study C case metadata.source_split/source_openr1_ids from study_c_gold/target_plans.json"
    )
    p.add_argument(
        "--study-c",
        type=str,
        default="data/openr1_psy_splits/study_c_test.json",
        help="Path to study_c_test.json",
    )
    p.add_argument(
        "--plans",
        type=str,
        default="data/study_c_gold/target_plans.json",
        help="Path to target_plans.json",
    )
    args = p.parse_args()

    study_c_path = Path(args.study_c)
    plans_path = Path(args.plans)

    study_payload = _load_json(study_c_path)
    plans_payload = _load_json(plans_path)

    cases: List[Dict[str, Any]] = study_payload.get("cases") or []
    plans: Dict[str, Any] = plans_payload.get("plans") or {}

    updated_cases = 0
    updated_fields = 0

    for case in cases:
        case_id = case.get("id")
        if not case_id:
            continue

        plan_entry = plans.get(case_id) or {}
        plan_source_split = plan_entry.get("source_split")
        plan_source_id = plan_entry.get("source_openr1_id")

        meta = case.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
            case["metadata"] = meta

        if plan_source_split in {"test", "train"} and plan_source_id is not None:
            desired_ids = [int(plan_source_id)]
            if meta.get("source_openr1_ids") != desired_ids:
                meta["source_openr1_ids"] = desired_ids
                updated_fields += 1
            if meta.get("source_split") != plan_source_split:
                meta["source_split"] = plan_source_split
                updated_fields += 1
        else:
            if meta.get("source_openr1_ids") != []:
                meta["source_openr1_ids"] = []
                updated_fields += 1
            if meta.get("source_split") != "generated":
                meta["source_split"] = "generated"
                updated_fields += 1

        updated_cases += 1

    study_payload["cases"] = cases
    _write_json(study_c_path, study_payload)

    print(f"updated_cases {updated_cases}")
    print(f"updated_fields {updated_fields}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
