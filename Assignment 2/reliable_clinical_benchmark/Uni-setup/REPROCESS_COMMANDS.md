# Commands to Reprocess Study A Data

After updating the refusal detection logic in `src/reliable_clinical_benchmark/metrics/extraction.py`, you need to reprocess all extracted data.

## Step 1: Reprocess All Models

Run the extraction script to regenerate all processed files with the updated refusal detection logic:

```powershell
cd "E:\22837352\NLP\NLP-Module\Assignment 2\reliable_clinical_benchmark\Uni-setup"
& "D:\Anaconda3\Scripts\activate" mh-llm-local-env
$Env:PYTHONNOUSERSITE="1"
$Env:PYTHONPATH="src"
python scripts\study_a\metrics\extract_predictions.py
```

**What it does:**
- Reads all `results/{model-id}/study_a_generations.jsonl` files
- Applies the updated refusal detection logic (context-aware, diagnosis-first)
- Writes updated `processed/study_a_extracted/{model-id}/study_a_extracted.jsonl` files

**Expected output:**
- Processing messages for each model (e.g., "Processing qwq...", "Processing psyllm-gml-local...")
- No errors

## Step 2: Regenerate Validation Report (Optional)

After reprocessing, regenerate the validation report to see updated statistics:

```powershell
python scripts\study_a\metrics\generate_validation_report.py
```

**Output:** `processed/study_a_extracted/VALIDATION_REPORT.md`

## Step 3: Verify Processing

Check that processed files were updated:

```powershell
# Check file timestamps
Get-ChildItem processed\study_a_extracted\*\study_a_extracted.jsonl | Select-Object FullName, LastWriteTime
```

## What Changed

The updated extraction logic:
1. **Context-aware refusal detection**: Disclaimers at the end of responses with valid diagnoses are NOT flagged as refusals
2. **Diagnosis-first extraction**: Diagnoses are extracted BEFORE refusal checking
3. **Improved pattern matching**: Better detection of "consult a licensed healthcare professional" and similar phrases

All processed files will now reflect these improvements.

