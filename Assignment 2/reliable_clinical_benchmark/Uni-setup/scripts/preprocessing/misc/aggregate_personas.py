import json
import os
from pathlib import Path

def get_age_from_band(band):
    """Derives a single age from a band string."""
    if not band:
        return 25
    band = str(band).strip()
    if "+" in band:
        try:
            return int(band.replace("+", "").strip()) + 2
        except:
            return 78
    if "-" in band:
        parts = band.split("-")
        try:
            return (int(parts[0].strip()) + int(parts[1].strip())) // 2
        except:
            return 25
    try:
        return int(band)
    except ValueError:
        return 25

def aggregate_personas():
    base_dir = Path("e:/22837352/NLP/NLP-Module/Assignment 2/docs/personas")
    output_file = Path("e:/22837352/NLP/NLP-Module/Assignment 2/docs/patient template/personas.json")
    registry_file = base_dir / "persona_registry_v2.json"
    
    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    risk_map = {p["id"]: p.get("risk_level", "low") for p in registry}
    condition_map = {p["id"]: p.get("condition", "") for p in registry}
    
    personas_list = []
    
    # Iterate through directories
    for persona_dir in base_dir.iterdir():
        if not persona_dir.is_dir():
            continue
            
        patient_file = persona_dir / "patient.json"
        messages_file = persona_dir / "messages.json"
        
        if not patient_file.exists() or not messages_file.exists():
            continue
            
        with open(patient_file, "r", encoding="utf-8") as f:
            patient_data = json.load(f)
        with open(messages_file, "r", encoding="utf-8") as f:
            messages_data = json.load(f)
            
        p_id = patient_data["id"]
        info = patient_data.get("patient_info", {})
        demo = info.get("demographics", {})
        acc = info.get("accessibility", {})
        context_data = info.get("self_reported_context", {})
        
        # Mapping logic
        persona_entry = {
            "id": p_id,
            "name": patient_data.get("name", p_id.capitalize()),
            "age": get_age_from_band(demo.get("age_band", "21")),
            "pronouns": demo.get("pronouns", "they/them"),
            "tone": acc.get("preferred_tone", "gentle"),
            "risk": risk_map.get(p_id, "low"),
            "context": f"{condition_map.get(p_id, '')}. {context_data.get('main_struggles', '')}",
            "messages": messages_data
        }
        
        personas_list.append(persona_entry)
    
    # Sort for consistency
    personas_list.sort(key=lambda x: x["id"])
    
    output_data = {"personas": personas_list}
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully aggregated {len(personas_list)} personas into {output_file}")

if __name__ == "__main__":
    aggregate_personas()
