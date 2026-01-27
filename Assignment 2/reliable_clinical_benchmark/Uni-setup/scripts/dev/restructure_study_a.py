from pathlib import Path
import shutil
import os

def restructure_study_a():
    base_dir = Path("metric-results/study_a")
    
    # Get all metrics files except all_models_metrics.json
    files = list(base_dir.glob("*_metrics.json"))
    
    for file in files:
        if file.name == "all_models_metrics.json":
            continue
            
        model_name = file.name.replace("_metrics.json", "")
        print(f"Processing {model_name}...")
        
        # Create model directory
        model_dir = base_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Move and rename file
        dest = model_dir / "study_a_metrics.json"
        shutil.move(file, dest)
        print(f"Moved {file} -> {dest}")

if __name__ == "__main__":
    restructure_study_a()

