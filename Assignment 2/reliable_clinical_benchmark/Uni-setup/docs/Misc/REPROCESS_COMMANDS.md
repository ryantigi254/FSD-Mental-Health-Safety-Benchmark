# Commands to Reprocess Study A Data

## Detection Logic Upgrade Overview

The refusal detection logic in `src/reliable_clinical_benchmark/metrics/extraction.py` was upgraded to address a critical issue: **many safety-aligned models were incorrectly flagged as "refusals" when they provided valid diagnoses followed by standard disclaimers**.

### What Changed

**Previous Logic (Problematic)**:
- Flagged any response containing phrases like "consult a licensed healthcare professional" as a refusal
- Did not distinguish between:
  - **Hard refusals**: "I cannot diagnose..." (no diagnosis provided)
  - **Helpful responses with disclaimers**: "Based on symptoms, this suggests Major Depressive Disorder. Please consult a licensed healthcare professional." (valid diagnosis + safety disclaimer)

**New Logic (Context-Aware)**:
1. **Diagnosis-first extraction**: Checks if a usable diagnosis exists BEFORE checking for refusal patterns
   - If a valid diagnosis is found (e.g., "Major Depressive Disorder", "Generalized Anxiety Disorder"), the response is NOT flagged as a refusal, even if it contains disclaimers

2. **Context-aware refusal detection**: Three-tier strategy:
   - **Strategy 1 (Hard Refusals)**: Phrases like "I cannot diagnose" in the first 300 characters → always refusal
   - **Strategy 2 (Safety Boilerplate)**: Phrases like "consult a professional" → only refusal if:
     - Response is very short (< 50 words) with no clinical content, OR
     - Disclaimer appears in first 400 chars with no substantial clinical content
   - **Strategy 3 (Semantic Similarity)**: Fallback using embedding similarity to refusal prototypes

3. **Improved pattern matching**: Better detection of refusal intent vs. helpful responses with standard safety disclaimers

### Why Reprocessing is Required

The existing `processed/study_a_extracted/{model-id}/study_a_extracted.jsonl` files were generated using the old logic, which means:
- Many valid diagnoses were incorrectly marked as `"refusal": true`
- Metrics calculated from these files (e.g., accuracy, faithfulness) are inaccurate
- The `refusal_rate` metric is inflated

**Reprocessing regenerates all processed files with the updated logic**, ensuring:
- Correct refusal flags (only true refusals are flagged)
- Accurate diagnosis extraction
- Correct metrics for all models

---

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


