
import json
import sys
from pathlib import Path
from collections import Counter

def calculate_repetition_score(text: str) -> float:
    """
    Calculate a simple repetition score.
    Higher score = more repetitive.
    Returns: (1 - unique_lines / total_lines)
    """
    if not text: 
        return 0.0
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    unique_lines = set(lines)
    return 1.0 - (len(unique_lines) / len(lines))

def main():
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / "results"
    cleaned_dir = base_dir / "processed" / "study_a_cleaned"
    
    print("=" * 60)
    print("VERIFYING REPETITION REMOVAL")
    print("=" * 60)
    
    models = [d.name for d in cleaned_dir.iterdir() if d.is_dir()]
    
    worst_offender_sample = None
    max_len_reduction = 0
    
    for model in models:
        raw_file = raw_dir / model / "study_a_generations.jsonl"
        cleaned_file = cleaned_dir / model / "study_a_generations.jsonl"
        
        if not raw_file.exists():
            print(f"Skipping {model} (No raw file)")
            continue
            
    with open("repetition_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("REPETITION REMOVAL REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        found_any = False
        for model in models:
            raw_file = raw_dir / model / "study_a_generations.jsonl"
            cleaned_file = cleaned_dir / model / "study_a_generations.jsonl"
            
            if not raw_file.exists(): continue
                
            with open(raw_file, 'r', encoding='utf-8') as rf:
                raw_data = {json.loads(l)['id']: json.loads(l) for l in rf if l.strip()}
            with open(cleaned_file, 'r', encoding='utf-8') as cf:
                cleaned_data = {json.loads(l)['id']: json.loads(l) for l in cf if l.strip()}
                
            fixes = 0
            removed_chars = 0
            example_sid = None
            
            for sid, entry in cleaned_data.items():
                if sid not in raw_data: continue
                r_txt = raw_data[sid].get('output_text', '')
                c_txt = entry.get('output_text', '')
                
                diff = len(r_txt) - len(c_txt)
                if diff > 1000:
                    fixes += 1
                    removed_chars += diff
                    if not example_sid or diff > (len(raw_data[example_sid]['output_text']) - len(cleaned_data[example_sid]['output_text'])):
                        example_sid = sid
            
            if fixes > 0:
                found_any = True
                f.write(f"MODEL: {model}\n")
                f.write(f"  - Repetition Fixes: {fixes} samples\n")
                f.write(f"  - Total Chars Removed: {removed_chars}\n")
                if example_sid:
                    r = raw_data[example_sid]['output_text']
                    c = cleaned_data[example_sid]['output_text']
                    f.write(f"  - Example ({example_sid}): {len(r)} -> {len(c)} chars\n")
                    f.write(f"    Raw End: ...{r[-100:].replace(chr(10), ' ')}\n")
                    f.write("\n")

        if not found_any:
            f.write("No significant repetition removal detected in any model.\n")
            
    print("Report written to repetition_report.txt")

if __name__ == "__main__":
    main()


