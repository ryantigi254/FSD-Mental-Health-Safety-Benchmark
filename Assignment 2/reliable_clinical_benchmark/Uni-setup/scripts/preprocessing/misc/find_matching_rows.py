import json
import re
from pathlib import Path

def find_matches():
    path = Path("data/openr1_psy_splits/study_a_test.json")
    print(f"Loading {path}...")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            samples = data.get("samples", [])
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    # Define keywords for each persona
    personas = {
        "aisha": ["depression", "depressive", "sertraline", "tired", "bedroom"],
        "sam": ["ptsd", "post-traumatic", "nightmares", "prazosin", "assault"],
        "noor": ["grief", "loss", "father", "mirtazapine", "ibuprofen"],
        "jamal": ["panic", "heart", "train", "sertraline", "propranolol"],
        "kai": ["autism", "sensory", "loud", "peanut", "school"],
        "priya": ["social anxiety", "presentation", "sertraline", "latex"],
        "tomas": ["anger", "personality", "blood pressure", "ibuprofen"],
        "leo": ["adhd", "attention", "focus", "lisdexamfetamine", "university"],
        "eleni": ["ocd", "obsessive", "baby", "harm", "checking"],
        "maya": ["borderline", "personality", "mood", "quetiapine", "abandonment"]
    }

    best_matches = {}

    print(f"Scanning {len(samples)} samples from Study A (Proxy for OpenR1)...")

    for pid, keywords in personas.items():
        best_score = 0
        best_idx = -1
        best_id_str = ""
        best_reasoning = ""

        for idx, sample in enumerate(samples):
            # Study A has 'prompt', 'gold_answer', 'gold_reasoning'
            # Convert reasoning list to string
            reasoning = " ".join(sample.get("gold_reasoning", []))
            text = (
                str(sample.get("prompt", "")) + " " + 
                str(sample.get("gold_answer", "")) + " " +
                reasoning
            ).lower()

            score = 0
            for kw in keywords:
                if kw in text:
                    score += 1
            
            # Penalize brevity
            if len(text) < 200:
                score = 0

            if score > best_score:
                best_score = score
                best_idx = idx
                best_id_str = sample.get("id") # a_001
                best_reasoning = reasoning[:200]

        best_matches[pid] = {
            "index": best_idx, 
            "id_str": best_id_str, 
            "score": best_score,
            "snippet": best_reasoning
        }
        print(f"Found match for {pid}: Index {best_idx} ({best_id_str}) (Score {best_score})")

    print("\n--- RESULTS ---")
    print("Paste these IDs into build_splits.py (using Index as OpenR1 ID proxy):")
    for pid, data in best_matches.items():
        # OpenR1 ID is roughly the index in Study A
        print(f"'{pid}': [{data['index']}],  # Score: {data['score']} ({data['id_str']})")

if __name__ == "__main__":
    find_matches()
