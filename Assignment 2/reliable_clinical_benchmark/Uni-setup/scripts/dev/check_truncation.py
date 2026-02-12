
import json
import sys
from pathlib import Path
import statistics

def check_file(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    print(f"\nAnalyzing: {path}")
    
    lengths = []
    errors = 0
    truncated_json = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                output = data.get("output_text", "")
                lengths.append(len(output))
            except json.JSONDecodeError:
                print(f"  ❌ Line {i}: Invalid JSON (potential truncation)")
                truncated_json += 1
                errors += 1

    if not lengths:
        print("  ⚠️ No valid entries found.")
        return

    max_len = max(lengths)
    avg_len = statistics.mean(lengths)
    
    print(f"  Entries: {len(lengths)}")
    print(f"  Max Length: {max_len}")
    print(f"  Avg Length: {avg_len:.0f}")
    
    # Heuristic for 16k limit (approx chars might vary, but ~16k chars is suspicious if exact)
    # 16384 tokens is much more than 16k chars, usually ~50-60k chars. 
    # But let's seeing if max length is huge.
    
    if truncated_json == 0:
        print("  ✅ All lines are valid JSON.")
    else:
        print(f"  ❌ {truncated_json} lines with invalid JSON.")

    print("\n  Top 5 Longest Endings:")
    # Sort by length descending, get top 5
    top_long = sorted([(len(d.get("output_text", "")), d.get("output_text", "")) for d in [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]], key=lambda x: x[0], reverse=True)[:5]
    
    for length, text in top_long:
        suffix = text[-50:].replace("\n", "\\n")
        print(f"    - [{length} chars] ...{suffix}")

files = [
    r"results/deepseek-r1-lmstudio/study_a_generations.jsonl",
    r"results/gpt-oss-20b/study_a_generations.jsonl",
    r"results/qwen3-lmstudio/study_a_generations.jsonl",
    r"results/qwq/study_a_generations.jsonl"
]

uni_setup = Path(r"e:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup")

for f in files:
    full_path = uni_setup / f
    check_file(full_path)
