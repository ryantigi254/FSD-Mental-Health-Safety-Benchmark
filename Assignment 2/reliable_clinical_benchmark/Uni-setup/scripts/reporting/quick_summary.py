"""Quick summary of cleaning results."""
import json
from pathlib import Path

base = Path(__file__).parent.parent.parent
models = ['deepseek-r1-lmstudio', 'gpt-oss-20b', 'piaget-8b-local', 
          'psych-qwen-32b-local', 'psyche-r1-local', 'psyllm-gml-local', 
          'qwen3-lmstudio', 'qwq']

print("OPTIMIZED CLEANING SUMMARY")
print("=" * 60)
print(f"{'Model':<26} {'Raw Chars':>12} {'Opt Chars':>12} {'Removed':>10}")
print("-" * 60)

total_raw = 0
total_opt = 0

for m in models:
    raw_path = base / 'results' / m / 'study_a_generations.jsonl'
    opt_path = base / 'processed' / 'study_a_cleaned_optimized' / m / 'study_a_generations.jsonl'
    if not raw_path.exists() or not opt_path.exists():
        continue
    
    raw_chars = sum(len(json.loads(l).get('output_text','')) 
                    for l in open(raw_path, encoding='utf-8') if l.strip())
    opt_chars = sum(len(json.loads(l).get('output_text','')) 
                    for l in open(opt_path, encoding='utf-8') if l.strip())
    
    total_raw += raw_chars
    total_opt += opt_chars
    pct = (raw_chars - opt_chars) / raw_chars * 100 if raw_chars > 0 else 0
    print(f"{m:<26} {raw_chars:>12,} {opt_chars:>12,} {pct:>9.1f}%")

print("-" * 60)
total_pct = (total_raw - total_opt) / total_raw * 100 if total_raw > 0 else 0
print(f"{'TOTAL':<26} {total_raw:>12,} {total_opt:>12,} {total_pct:>9.1f}%")
print("=" * 60)
print(f"\nTotal characters removed: {total_raw - total_opt:,}")
print(f"Reduction: {total_pct:.1f}%")

