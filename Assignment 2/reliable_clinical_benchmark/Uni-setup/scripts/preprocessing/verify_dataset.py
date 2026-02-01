import json
import os

def check_file(path, label):
    if not os.path.exists(path):
        print(f"{label}: MISSING")
        return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"{label}: {len(data)} items")
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k in data:
            val = data[k]
            if isinstance(val, list):
                print(f"Key '{k}': List with {len(val)} items")
    
    if isinstance(data, list) and len(data) > 0:
        print(f"Item 0 type: {type(data[0])}")
        if isinstance(data[0], dict):
             print(f"Item 0 keys: {list(data[0].keys())}")
        else:
             print(f"Item 0 content: {data[0]}")
    
    # Check for multi-turn (assuming 'data' or similar key if dict)
    items = data
    if isinstance(data, dict):
        # Heuristic: find the list
        for k in data:
            if isinstance(data[k], list):
                items = data[k]
                break
    
    if isinstance(items, list) and all(isinstance(x, dict) for x in items):
        multi = [x for x in items if "turns" in x and len(x["turns"]) > 1]
        if multi:
            print(f"  - Multi-turn items: {len(multi)}")
            print(f"  - Avg turns: {sum(len(x['turns']) for x in multi)/len(multi)}")
            print(f"  - First ID: {multi[0].get('id')}")
        
        # Check for source distribution
        sources = {}
        for x in items:
            s = x.get("metadata", {}).get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        print(f"  - Sources: {sources}")

check_file('data/openr1_psy_splits/study_b_test.json', 'Study B Single')
check_file('data/openr1_psy_splits/study_b_multi_turn_test.json', 'Study B Multi')
check_file('data/openr1_psy_splits/study_c_test.json', 'Study C')
