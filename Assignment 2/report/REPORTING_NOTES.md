# Reporting Notes (for write-up)

Purpose: capture \textbf{reporting/interpretation issues} discovered during implementation and early runs, so the final report can state limitations, controls, and any mitigation clearly.

## Scope
- These notes are \textbf{not} benchmark objectives.
- They are primarily about \textbf{interpretation, presentation, and reproducibility} of results and exemplars.

## Current reporting/interpretation issues to track

### 1) Persona traceability gaps in caches
- **Issue**: `persona_id` is not consistently present in current caches (in multiple files it is either missing entirely, or present-but-empty).
- **Impact on reporting**:
  - Persona-level audits and stratified reporting (by persona) are not currently reliable.
  - Any persona-stratified results must be labelled as provisional unless caches are regenerated or repaired with a trusted mapping.
- **Mitigation**:
  - Prefer regeneration so `persona_id` is always present and non-empty.
  - Only repair caches if there is a deterministic, trustworthy mapping from row ID to persona.

### 2) Output-format variability and hidden-reasoning scaffolding
- **Issue**: outputs include internal tags (e.g., `<think>`) and vary in whether they follow the requested schema markers (e.g., `REASONING:` / `DIAGNOSIS:`).
- **Impact on reporting**:
  - Changes what is countable as “reasoning” vs “answer”.
  - Can break parsers and bias metrics if extraction silently fails.
  - Affects qualitative exemplars (what the reader sees vs what the model “used”).
- **Mitigation**:
  - Enforce stable parsing and explicit failure logging.
  - Keep a clear policy for whether hidden reasoning tags are removed for display in the report.

### 3) Locale-specific signposting (not a benchmark objective)
- **Issue**: some generations include locale-specific crisis resources (e.g., US-only hotline references such as “988” and “U.S.”).
- **Why it matters**:
  - The benchmark’s goal is clinical behaviour under failure modes, not country-specific service navigation.
  - However, embedding US-only guidance in exemplars can distract reviewers/readers and complicate interpretation if personas are UK-oriented.
- **Mitigation**:
  - Document as a reporting/interpretation issue.
  - Only add filtering/localisation if needed for publication-quality exemplars.

### 4) Clinical framing references (e.g., DSM-5) in outputs
- **Issue**: some outputs frequently cite formal criteria (e.g., “DSM-5”).
- **Impact on reporting**:
  - Can inflate verbosity and shift outputs toward non-local clinical framing.
  - Can make exemplar selection look like “best-case” interpretability if these appear more often in some models than others.
- **Mitigation**:
  - Note prevalence differences when selecting exemplars.
  - Prefer exemplars that demonstrate the targeted failure mode rather than “citation/authority style”.

### 5) Status filtering and failure handling
- **Issue**: generations can include failures or malformed rows; report quality depends on strict filtering (e.g., `status=ok`) and transparent exclusion rules.
- **Impact on reporting**:
  - Without consistent filtering rules, reported metrics and exemplars can be biased.
- **Mitigation**:
  - Define a single inclusion policy (“use only `status=ok`”) and apply it uniformly across all studies.

## Where this is referenced
- `Assignment 2/docs/supervisor/Supervisor Briefing.tex`

