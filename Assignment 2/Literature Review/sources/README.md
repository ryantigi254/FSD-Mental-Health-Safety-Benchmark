# Literature Review Sources: Reading Organization

This directory contains the prioritized reading list and folder structure for systematically reviewing all papers cited in the literature review.

## Quick Stats

**Total Papers**: 60
- **Tier 1 (Must Read)**: 18 papers (30%)
- **Tier 2 (Important)**: 20 papers (33%)
- **Tier 3 (Reference)**: 22 papers (37%)

**Downloaded PDFs (local)**: 12
- Currently located under: `sources/<bucket>/<tier>/*.pdf`
- Note: PDFs under `sources/**` are gitignored by design (see `.gitignore`), so they will not be committed.

## Folder Structure

```
sources/
├── README.md (this file)
├── PRIORITY_READING_LIST.csv (prioritized list with scores)
├── bucket_a_faithfulness/     # Study A: Faithfulness Gap
│   ├── tier_1_must_read/     # Priority score ≥ 4.5 (4 papers)
│   ├── tier_2_important/     # Priority score 4.0-4.49 (4 papers)
│   └── tier_3_reference/      # Priority score < 4.0 (3 papers)
├── bucket_b_sycophancy/       # Study B: Sycophancy
│   ├── tier_1_must_read/     # Priority ≥ 4.5 (7 papers)
│   ├── tier_2_important/     # Priority 4.0-4.49 (6 papers)
│   └── tier_3_reference/      # Priority < 4.0 (6 papers)
├── bucket_c_silent_bias/      # Study A: Silent Bias Rate (part of Study A)
│   ├── tier_1_must_read/     # Priority ≥ 4.5 (0 papers*)
│   ├── tier_2_important/     # Priority 4.0-4.49 (2 papers)
│   └── tier_3_reference/      # Priority < 4.0 (0 papers)
├── bucket_d_longitudinal/    # Study C: Longitudinal Drift
│   ├── tier_1_must_read/     # Priority ≥ 4.5 (3 papers)
│   ├── tier_2_important/     # Priority 4.0-4.49 (2 papers)
│   └── tier_3_reference/      # Priority < 4.0 (2 papers)
├── clinical_domain/            # Clinical/Mental Health domain papers
│   ├── tier_1_must_read/     # Priority ≥ 4.5 (3 papers)
│   ├── tier_2_important/     # Priority 4.0-4.49 (4 papers)
│   └── tier_3_reference/      # Priority < 4.0 (0 papers)
└── evaluation_tools/          # Evaluation instruments & frameworks
    ├── tier_1_must_read/     # Priority ≥ 4.5 (1 paper)
    ├── tier_2_important/     # Priority 4.0-4.49 (2 papers)
    └── tier_3_reference/      # Priority < 4.0 (1 paper)
```

*Note: Bucket C papers are primarily in Clinical Domain category

## Prioritization System

### Scoring Formula

Priority Score = 0.30×Directness + 0.20×Evidence_Quality + 0.15×Transferability + 0.10×Recency + 0.15×Bucket_Relevance + 0.10×Implementation_Value

**Where:**
- **Directness (1-5)**: How directly the paper addresses the specific failure mode
- **Evidence Quality (1-5)**: Rigor of methodology (RCT > Observational > Framework > Review)
- **Transferability (1-5)**: How applicable findings are to our clinical LLM context
- **Recency (1-5)**: Publication year (2025=5, 2024=4, 2023=3, etc.)
- **Bucket Relevance (1-5)**: Relevance to the specific bucket's failure mode
- **Implementation Value (1-5)**: How useful for implementing our evaluation framework

### Tier Assignment

- **Tier 1 (Must Read)**: Priority Score ≥ 4.5
  - Core foundational papers
  - Papers with quantitative metrics we directly use
  - Papers introducing key methodologies
  
- **Tier 2 (Important)**: Priority Score 4.0-4.49
  - Supporting evidence
  - Papers with relevant findings but less direct application
  - Complementary methodologies

- **Tier 3 (Reference)**: Priority Score < 4.0
  - Background context
  - Papers cited but not central to implementation
  - Can be scanned/skimmed rather than fully read

## Tier 1 Papers by Bucket (Must Read - Priority ≥ 4.5)

### Bucket A: Faithfulness Gap (4 papers)
1. **Lanham et al. (2023)** - "Measuring Faithfulness in Chain-of-Thought Reasoning" (4.65)
   - Foundational methodology; Early Answering protocol
2. **Lanham et al. (2024)** - "Making Reasoning Matter" (4.75)
   - Updated framework; FRODO methodology
3. **Turpin et al. (2023)** - "Language Models Don't Always Say What They Think" (4.65)
   - 36% accuracy drop; biasing features study
4. **Paul et al. (2024)** - "Making Reasoning Matter" (4.75)
   - FRODO framework introduction

### Bucket B: Sycophancy (7 papers)
1. **Fanous et al. (2025)** - "SycEval: Evaluating LLM Sycophancy" (5.0)
   - Clinical context; progressive/regressive rates
2. **Pandey et al. (2025)** - "Beacon: Single-Turn Diagnosis and Mitigation" (5.0)
   - Latent sycophancy detection
3. **Liu et al. (2025)** - "Truth Decay: Quantifying Multi-Turn Sycophancy" (5.0)
   - Multi-turn quantification
4. **Kaur (2025)** - "Echoes of Agreement: Argument-Driven Sycophancy" (4.85)
   - Pylons of Agreement sequence
5. **Hong et al. (2025)** - "ELEPHANT and SYCON-Bench" (4.85)
   - Third-person framing reduces sycophancy 60%+
6. **Wei et al. (2023)** - "Simple Synthetic Data Reduces Sycophancy" (4.65)
   - RLHF increases sycophancy 18% to 75%
7. **Anthropic (2024)** - "Towards Understanding Sycophancy" (4.7)
   - Comprehensive analysis

### Bucket C: Silent Bias (2 papers) - **Part of Study A**
1. **Gabriel et al. (2024)** - "Can AI Relate" (4.3) - *Tier 2, but critical for bias*
   - 2-13% lower empathy for Black patients
2. **Lee et al. (2024)** - "SafeHear" (4.85)
   - 92% generic vs 68% clinical safety gap

**Note**: Silent Bias is evaluated as part of **Study A**, not a separate study. Study A includes both Faithfulness (Bucket A) and Silent Bias (Bucket C).

### Bucket D: Longitudinal Drift (3 papers)
1. **Laban et al. (2025)** - "LLMs Get Lost In Multi-Turn Conversation" (5.0)
   - 39% degradation; 90% to 65% drop
2. **Kruse et al. (2025)** - "Large Language Models with Temporal Reasoning" (5.0)
   - Clinical longitudinal; PDSQI-9 integration
3. **Zheng et al. (2024)** - "Why LLMs Fail in Multi-Turn Conversations" (4.7)
   - 39% accuracy drop analysis

### Clinical Domain (3 papers)
1. **Zhang et al. (2025) / PsyLLM** - "Beyond Empathy" (5.0)
   - **CRITICAL**: Our primary data source (OpenR1-Psy)
2. **Hager et al. (2024)** - "Evaluation and Mitigation of Limitations" (4.85)
   - Models fail in realistic workflows
3. **Lee et al. (2024)** - "SafeHear" (4.85)
   - Domain-specific safety gap

### Evaluation Tools (1 paper)
1. **Kim et al. (2025)** - "PDSQI-9 Development and Validation" (5.0)
   - **CRITICAL**: Used in Study C for quality measurement

## Highest Priority Papers (Score 5.0)

1. **Fanous et al. (2025)** - SycEval
2. **Pandey et al. (2025)** - Beacon
3. **Liu et al. (2025)** - Truth Decay
4. **Laban et al. (2025)** - Multi-turn drift
5. **Kruse et al. (2025)** - Clinical longitudinal
6. **Zhang et al. (2025)** - PsyLLM/OpenR1-Psy
7. **Kim et al. (2025)** - PDSQI-9

## Reading Order Recommendation

### Week 1: Core Foundations
1. **Turpin et al. (2023)** - Faithfulness Gap (Bucket A)
2. **Lanham et al. (2023)** - Faithfulness methodology (Bucket A)
3. **Wei et al. (2023)** - Sycophancy foundation (Bucket B)
4. **Laban et al. (2025)** - Longitudinal Drift (Bucket D)

### Week 2: Implementation Details
5. **Lanham et al. (2024)** - Updated faithfulness (Bucket A)
6. **Fanous et al. (2025)** - SycEval clinical sycophancy (Bucket B)
7. **Liu et al. (2025)** - Multi-turn sycophancy (Bucket B)
8. **Kruse et al. (2025)** - Clinical longitudinal (Bucket D)

### Week 3: Clinical Context & Data
9. **Zhang et al. (2025) / PsyLLM** - OpenR1-Psy dataset (Clinical)
10. **Hager et al. (2024)** - Clinical limitations (Clinical)
11. **Lee et al. (2024)** - SafeHear benchmark (Clinical/Bucket C)
12. **Kim et al. (2025)** - PDSQI-9 instrument (Tools)

### Week 4: Advanced Methods
13. **Paul et al. (2024)** - FRODO framework (Bucket A)
14. **Pandey et al. (2025)** - Beacon latent probe (Bucket B)
15. **Kaur (2025)** - Argument-driven sycophancy (Bucket B)
16. **Hong et al. (2025)** - SYCON-Bench (Bucket B)

### Week 5: Supporting Evidence
17. **Anthropic (2024)** - Sycophancy understanding (Bucket B)
18. **Zheng et al. (2024)** - Multi-turn failures (Bucket D)
19. **Gabriel et al. (2024)** - Bias in mental health (Bucket C)
20. **Tier 2 papers** - As needed for specific gaps

## Quick Start Guide

### Step 1: Review Prioritization
1. Open `PRIORITY_READING_LIST.csv`
2. Filter by `Tier = "Tier 1"` to see must-read papers
3. Check `PRIORITIZATION_SUMMARY.md` for reading order

### Step 2: Download Papers
1. Open `PRIORITY_READING_LIST.csv`
2. Filter by `Tier = "Tier 1"` and `Reading_Status = "Not Started"`
3. For each paper:
   - Click the `Primary_Link` to download
   - Save PDF to the appropriate folder: `{bucket}/{tier}/`
   - Name file: `{Author_Last_Name}_{Year}_{Short_Title}.pdf`
   - Example: `Turpin_2023_Language_Models_Dont_Always_Say.pdf`

### Step 3: Read & Track Progress
1. Read papers in the recommended order (Week 1-5 plan above)
2. After reading each paper:
   - Update `PRIORITY_READING_LIST.csv`:
     - Set `Reading_Status` to "Completed"
     - Add key takeaways in `Notes` column
     - Note any implementation relevance
3. Move to next Tier 1 paper

### Step 4: Tier 2 & 3
- **Tier 2**: Read after completing Tier 1 for each bucket
- **Tier 3**: Scan abstracts; read full text only if needed for specific details

## File Naming Convention

When saving papers, use this format:
```
{Last_Author_Name}_{Year}_{Short_Title}.pdf
```

**Examples:**
- `Turpin_2023_Language_Models_Dont_Always_Say.pdf`
- `Lanham_2023_Measuring_Faithfulness.pdf`
- `Wei_2023_Simple_Synthetic_Data_Reduces_Sycophancy.pdf`
- `Laban_2025_LLMs_Get_Lost_In_Multi_Turn_Conversation.pdf`
- `Zhang_2025_Beyond_Empathy_Integrating_Diagnostic.pdf`

## Strategy for Narrowing Down Sources

If you have limited time, use these quantitative filters:

### Filter 1: Tier 1 Only (18 papers)
**Priority Score ≥ 4.5** - Time estimate: 2-3 weeks

### Filter 2: Priority Score ≥ 5.0 (7 papers - CRITICAL)
**Time estimate**: 1 week for focused reading
1. Fanous et al. (2025) - SycEval
2. Pandey et al. (2025) - Beacon
3. Liu et al. (2025) - Truth Decay
4. Laban et al. (2025) - Multi-turn drift
5. Kruse et al. (2025) - Clinical longitudinal
6. Zhang et al. (2025) - PsyLLM/OpenR1-Psy
7. Kim et al. (2025) - PDSQI-9

### Filter 3: By Study Implementation Need
- **Study A (Faithfulness + Bias)**: 6 papers
- **Study B (Sycophancy)**: 5 papers
- **Study C (Longitudinal Drift)**: 4 papers

### Filter 4: Quantitative Metrics Focus
Papers providing specific quantitative findings we cite:
- 36% accuracy drop (Turpin et al., 2023)
- 18% to 75% sycophancy increase (Wei et al., 2023)
- 39% degradation (Laban et al., 2025; Zheng et al., 2024)
- 92% vs 68% safety gap (Lee et al., 2024)
- 43.52% progressive sycophancy (Fanous et al., 2025)
- 2-13% empathy gap (Gabriel et al., 2024)

### Recommended Narrowing Approach
- **1 week**: Read Filter 2 (7 papers, Priority ≥ 5.0)
- **2 weeks**: Read Filter 2 + Filter 3 (Study-specific papers, ~15 papers)
- **3-4 weeks**: Read Filter 1 (All Tier 1, 18 papers)
- **5+ weeks**: Read Tier 1 + Tier 2 (38 papers)

## Quick Decision Matrix

| Priority Score | Reading Depth | Time per Paper |
|---------------|---------------|----------------|
| ≥ 5.0 | Full read + notes | 2-3 hours |
| 4.5-4.9 | Full read | 1-2 hours |
| 4.0-4.4 | Read + key sections | 45-60 min |
| 3.5-3.9 | Abstract + key findings | 20-30 min |
| < 3.5 | Abstract scan only | 5-10 min |

## Reading Strategy

### Phase 1: Tier 1 Papers (Must Read)
1. Start with **Bucket A (Faithfulness)** - Core to Study A
2. Then **Bucket C (Silent Bias)** - Also part of Study A (bias component)
3. Then **Bucket B (Sycophancy)** - Core to Study B
4. Then **Bucket D (Longitudinal)** - Core to Study C
5. Finally **Clinical Domain** - Context and validation

### Phase 2: Tier 2 Papers (Important)
- Read after completing Tier 1 for each bucket
- Focus on papers that fill gaps or provide supporting evidence

### Phase 3: Tier 3 Papers (Reference)
- Scan abstracts and key findings
- Read full text only if needed for specific details
- Use as reference when writing

## Bucket Mapping

| Bucket | Failure Mode | Study | Key Metrics |
|--------|-------------|-------|-------------|
| **A** | Faithfulness Gap | **Study A** | Δ_Reasoning, Step-F1 |
| **B** | Sycophancy | **Study B** | P_Syc, H_Ev, Flip Rate |
| **C** | Silent Bias | **Study A** (bias component) | R_SB (Silent Bias Rate) |
| **D** | Longitudinal Drift | **Study C** | Entity Recall, K_Conflict |

**Important**: 
- **Study A** evaluates **both** Bucket A (Faithfulness Gap) and Bucket C (Silent Bias Rate)
- Bucket C is **not a separate study** - it's part of Study A
- Study B = Sycophancy only
- Study C = Longitudinal Drift only

## Paper Count by Bucket & Tier

| Bucket | Tier 1 | Tier 2 | Tier 3 | Total |
|--------|--------|--------|--------|-------|
| **A (Faithfulness)** | 4 | 4 | 3 | 11 |
| **B (Sycophancy)** | 7 | 6 | 6 | 19 |
| **C (Silent Bias)** | 0 | 2 | 0 | 2 |
| **D (Longitudinal)** | 3 | 2 | 2 | 7 |
| **Clinical Domain** | 3 | 4 | 0 | 7 |
| **Tools** | 1 | 2 | 1 | 4 |
| **Methods** | 0 | 0 | 2 | 2 |
| **Surveys** | 0 | 1 | 1 | 2 |
| **Resources** | 0 | 0 | 5 | 5 |
| **TOTAL** | **18** | **21** | **20** | **59** |

## Implementation-Focused Reading

When reading, focus on extracting:
1. **Quantitative metrics** (percentages, rates, scores)
2. **Methodology details** (protocols, algorithms, formulas)
3. **Implementation guidance** (pseudocode, frameworks, tools)
4. **Limitations** (what doesn't work, gaps identified)
5. **Clinical context** (how findings apply to mental health)

Skip or skim:
- Background/literature review sections (unless seminal)
- Detailed related work (unless directly relevant)
- Extensive experimental setup (unless implementing similar)
- Future work sections (unless critical for our direction)

## Notes

- All papers are from the Literature Review README bibliography
- Papers may appear in multiple buckets if relevant to multiple failure modes
- Priority scores are based on relevance to our specific evaluation framework
- Update `PRIORITY_READING_LIST.csv` as you read (add notes, completion status)
