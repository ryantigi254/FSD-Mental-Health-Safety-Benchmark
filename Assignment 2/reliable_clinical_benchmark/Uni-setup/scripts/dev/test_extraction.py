
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from reliable_clinical_benchmark.metrics.extraction import is_refusal, clean_model_output

text = "<think>Some reasoning...</think>\n<content>Attachment-related challenges impacting trust and relational safety</content>"
cleaned = clean_model_output(text)
print(f"Cleaned: '{cleaned}'")
print(f"Is Refusal: {is_refusal(text)}")

