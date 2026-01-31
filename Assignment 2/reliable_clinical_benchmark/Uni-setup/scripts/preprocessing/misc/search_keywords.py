import json
from pathlib import Path

def search():
    path = Path("data/openr1_psy_splits/study_a_test.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        samples = data.get("samples", [])

    keywords = ["check", "ritual", "abandon", "relationship", "break up"]
    
    for kw in keywords:
        print(f"\n--- Searching for '{kw}' ---")
        found = False
        for idx, s in enumerate(samples):
            text = (str(s.get("prompt", "")) + " " + str(s.get("gold_reasoning", ""))).lower()
            if kw in text:
                print(f"Match at Index {idx} ({s['id']}): {text[:100]}...")
                found = True
        if not found:
            print("No match found.")

if __name__ == "__main__":
    search()
