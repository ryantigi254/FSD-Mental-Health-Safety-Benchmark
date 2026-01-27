
import os
import sys

# Add src to sys.path
sys.path.append(os.path.abspath("src"))

try:
    import torch
    import transformers
    import spacy
    import scispacy
    from sentence_transformers import SentenceTransformer
    import datasets
    
    print("Core libraries imported successfully.")
    
    # Test scispaCy
    nlp = spacy.load("en_core_sci_sm")
    print("scispaCy model loaded successfully.")
    
    # Test project imports
    from reliable_clinical_benchmark.metrics.drift import MedicalNER
    ner = MedicalNER()
    print("Project MedicalNER initialized successfully.")
    
    from reliable_clinical_benchmark.utils.nli import NLIModel
    # nli = NLIModel() # Might attempt to download model, skip for import test only
    print("Project NLIModel module imported successfully.")
    
    print("All basic module tests passed.")
    
except Exception as e:
    print(f"Test failed: {e}")
    sys.exit(1)

