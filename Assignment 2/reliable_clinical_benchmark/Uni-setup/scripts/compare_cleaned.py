from pathlib import Path
import json

orig = Path("results/deepseek-r1-lmstudio/study_c_generations.jsonl")
proc = Path("processed/misc/deepseek-r1-lmstudio_study_c_generations.cleaned.jsonl")

changed = 0
same = 0
with orig.open("r", encoding="utf-8") as fo, proc.open("r", encoding="utf-8") as fp:
    for lo, lp in zip(fo, fp):
        if not lo.strip() or not lp.strip():
            continue
        o = json.loads(lo)
        p = json.loads(lp)
        ot = o.get("response_text") or o.get("output_text") or ""
        pt = p.get("response_text") or p.get("output_text") or ""
        if ot != pt:
            changed += 1
        else:
            same += 1

print(f"Same: {same}")
print(f"Changed: {changed}")
