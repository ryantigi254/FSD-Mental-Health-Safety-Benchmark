# Complete Literature Review Cycle for AI Paper Writing (with Code Development)

**Target Audience:** University-level AI/ML researcher building a publishable paper alongside code implementation  
**Foundation:** University of Northampton guidelines + Sheffield + best practices from peer institutions  
**Focus:** Efficient literature review that feeds into high-quality paper writing without overwhelming your coding timeline

---

## Phase 1: Planning & Scoping (Weeks 1-2)

### 1.1 Define Your Research Question Clearly

Before you search, crystallize your research question into a single sentence. This is your **North Star**.

**Example for AI/ML domain:**
- ❌ Bad: "How do LLMs work?"
- ✅ Good: "How does sycophancy in LLMs affect faithfulness in clinical reasoning tasks?"

**Action:** Write your research question and break it down into **3–5 key concepts** (e.g., sycophancy, faithfulness, clinical reasoning, LLMs).

### 1.2 Extract Keywords & Search Phrases

From your research question, pull out meaningful search terms. For each concept, generate **synonyms and related terms**.

**Example breakdown:**
- **Concept 1 (Sycophancy):** sycophancy, agreement bias, telling-users-what-they-want, alignment faking, yes-saying, deception in LLMs
- **Concept 2 (Faithfulness):** faithfulness, interpretability, explainability, hallucination, factual accuracy, truthfulness
- **Concept 3 (Clinical reasoning):** clinical NLP, medical AI, diagnostic reasoning, clinical decision support, healthcare AI
- **Concept 4 (LLMs):** large language models, transformer models, foundation models, generative AI

**Create a searchable document** (Google Sheet or Notion) to track all keywords. Mark which ones you've tested and which databases yielded results.

### 1.3 Plan Your Search Strategy

**Boolean logic for narrowing/broadening:**
- **AND** narrows (makes results more specific): `sycophancy AND clinical reasoning AND LLMs`
- **OR** broadens (finds related terms): `sycophancy OR "alignment faking" OR "yes-saying"`
- **NOT** excludes unwanted results: `LLMs NOT training NOT architecture` (if you only want evaluation papers)

**Example search progression (ordered from broad to narrow):**
1. `large language models faithfulness` (broad overview)
2. `LLMs AND sycophancy` (narrower focus)
3. `sycophancy AND clinical` (your core topic)
4. `"alignment faking" AND LLMs AND medical` (very specific)

---

## Phase 2: Finding & Gathering Sources (Weeks 2-4)

### 2.1 Choose Your Databases (For AI/ML Papers)

**Primary databases for AI/NLP research:**
- **arXiv.org** (preprints, fastest for new research) — search via title/abstract
- **ACL Anthology** (ACL, EMNLP, NAACL conferences) — highly selective, peer-reviewed
- **Google Scholar** (broad, includes preprints + published) — good for citation tracking
- **DBLP Computer Science Bibliography** (conference papers indexed by authors/venues)
- **OpenReview.net** (ICLR, NeurIPS submissions + reviews) — see peer feedback
- **PubMed Central** (if your work touches medical NLP)
- **Your university library databases** (often institutional access to IEEE, Springer, etc.)

### 2.2 Start Your Search Iteratively

Run 3–5 search queries per week. **Record every search** (what you searched, which database, number of results, which you filtered to).

**Search record template (spreadsheet row):**
```
| Date | Query | Database | Results Found | Papers Selected | Notes |
|------|-------|----------|----------------|-----------------|-------|
| 2025-12-20 | "sycophancy" AND LLMs | arXiv | 47 | 12 | Focused on 2024–2025 papers |
```

**Pro tip for AI papers:** Sort results by **relevance** or **date (newest first)** because the field moves fast. Papers from 2–3 years ago may already be superseded.

### 2.3 Use Citation Tracking (Forward & Backward)

Once you find a **key paper** (one that's directly on your topic):
- **Backward citation tracking:** Look at its reference list. Download 2–3 cited papers that sound relevant.
- **Forward citation tracking:** Use Google Scholar or Semantic Scholar to see who cited this paper. Often newer papers that build on it are gold.

**Example:** You find Anthropic's 2024 "Alignment Faking" paper. Check:
- Its citations (papers about deception in LLMs)
- Who cited it (recent work on detecting alignment faking)

---

## Phase 3: First-Pass Filtering & Skimming (Week 3)

### 3.1 The SKIM-DECIDE Rule: When to Read, Skim, or Skip

**Most papers should be SKIMMED, not read deeply.** Use this decision tree:

#### STEP 1: Title + Abstract (2–3 minutes per paper)
**DECIDE:**
- ✅ **READ FULLY** if: directly answers your research question, published in top venue (ACL/NeurIPS/ICLR), or cited heavily by others
- ⏱️ **SKIM (methods + results)** if: tangentially related, recent preprint, novel dataset, or methodological innovation
- ❌ **SKIP** if: outdated, irrelevant field, or makes claims without evidence

**Red flags to skip:**
- Very old (>5 years for AI, unless foundational)
- Predatory/low-quality venue
- Methodology described too vaguely
- Results section missing or makes unsupported claims

#### STEP 2: For Papers You'll Read or Skim

**Skim protocol (15–20 minutes per paper):**
1. **Title + abstract** (1 min) — does it match your question?
2. **Introduction + Conclusion** (3 min) — what's the main claim?
3. **Results section** (5 min) — what did they find?
4. **Methods (if key)** (3 min) — how did they do it? Is it reproducible?
5. **Figures & Tables** (3 min) — do visualizations clarify findings?
6. **Skim references** (1 min) — spot any key citations you missed?

**Full read protocol (45–60 min for key papers):**
- Read all sections sequentially
- Highlight key findings, methods, assumptions
- Note any gaps or limitations the authors acknowledge
- Compare their findings to other papers you've read

### 3.2 Create a Tracking Spreadsheet (Your Literature Matrix)

**Columns to include (adjust to your domain):**

```
| Paper ID | Title | Authors | Year | Venue | Topic Tag | Method | Key Finding | Relevance (1-5) | Status (Skim/Read/Skip) | Notes | Citation |
|----------|-------|---------|------|-------|-----------|--------|-------------|-----------------|-------------------------|-------|----------|
| P001 | "Alignment Faking..." | Wei et al. | 2024 | arXiv | Deception | Fine-tuning test | LLMs can fake alignment | 5 | READ | Core paper | Wei et al. (2024) |
| P002 | "Sycophancy in MLLMs" | Pi et al. | 2025 | EMNLP | Sycophancy | Multi-modal eval | VLMs agree with user labels even when wrong | 4 | READ | Multimodal extension | Pi et al. (2025) |
| P003 | "LLM Faithfulness Survey" | Zhao et al. | 2025 | arXiv | Faithfulness | Literature review | Categorizes 50+ papers on LLM limitations | 4 | SKIM | Use for taxonomy | Zhao et al. (2025) |
```

**Maintain this spreadsheet throughout your research.** It becomes your **golden index** for writing.

---

## Phase 4: Deep Reading & Critical Note-Taking (Weeks 4-8)

### 4.1 Critical Reading Approach

For each paper you decide to **READ FULLY**, use the **SQ3R method** (Survey → Question → Read → Recite → Review):

1. **Survey (2 min):** Scan abstract, headings, and conclusion. What's the paper about?
2. **Question (2 min):** Write 3 questions this paper should answer:
   - What problem does it solve?
   - How does it compare to prior work?
   - What are its limitations?
3. **Read (25–40 min):** Read carefully, annotating (digital or physical).
4. **Recite (5 min):** Close the paper. Write a 3–4 sentence summary from memory.
5. **Review (3 min):** Reread your summary against the paper. Correct gaps.

### 4.2 How to Take Notes That Support Report Writing

**Use a structured note template** (Google Doc or markdown file per paper):

```markdown
# Paper Notes: [Title]
**Authors:** [Names]  
**Year:** [YYYY]  
**Venue:** [Conference/Journal]  
**Citation:** [Author (Year)] or [BibTeX]  

## One-Sentence Summary
[Your single-sentence takeaway. This becomes a bullet point in your report.]

## Main Contribution
- [What's new here? Is it methodology, dataset, finding, or theory?]

## Key Findings
1. [Finding 1 with quote or figure reference if important]
2. [Finding 2]
3. [Finding 3]

## Methodology
- **Setup:** [What problem did they tackle?]
- **Data/Models:** [What data? Which LLM/baseline?]
- **Metrics:** [How did they measure success?]
- **Results:** [Quantitative outcomes. Copy table if important.]

## Critical Analysis (Your Voice)
- **Strengths:** What makes this paper strong?
- **Weaknesses:** Limitations? Unaddressed questions? Small sample size? Single domain?
- **Relevance to My Work:** How does this inform my research?
- **Disagreement (if any):** Do I think their conclusion holds? Why/why not?

## Connections to Other Papers
- [Related to Paper X because both study sycophancy]
- [Extends/Challenges Paper Y findings]

## Quotes Worth Citing
> "Direct quote that captures a key claim" — Page [X]

[Add 2–3 impactful quotes max. More = lazy note-taking.]

## TODO/Questions for Later
- [ ] Find the code/data they mention in footnote 3
- [ ] Check if Sheffield preprint differs from published version
- [ ] Verify statistical significance claim in Table 2
```

**Why this template works:**
- ✅ Captures findings *and* your critical voice (both needed for a report)
- ✅ Forces you to articulate *why* a paper matters to *you*
- ✅ Makes writing your report faster (copy sections directly)
- ✅ Helps you spot connections between papers (see "Connections" field)

### 4.3 Develop a Citation Management System

**Options:**
1. **Zotero** (free, powerful, integrates with Word/Google Docs)
2. **Mendeley** (free version exists, good for organizing PDFs)
3. **BibTeX + Overleaf** (if writing in LaTeX; best for technical papers)
4. **Notion/Roam Research** (if you like linked notes and want everything in one place)

**Minimum requirement:** Export all citations in your final format (BibTeX, APA, IEEE, etc.) so you don't manually type them later.

---

## Phase 5: Identifying Key Points & Making Connections (Weeks 5-8)

### 5.1 Build Your Synthesis Matrix

Once you have 15–25 papers read/skimmed, create a **Synthesis Matrix** to identify **themes** and **connections** across papers.

**Columns:**
```
| Theme | Paper ID | Key Quote/Finding | Agreement? (Y/N) | How it Connects | My Interpretation |
|-------|----------|-------------------|------------------|-----------------|-------------------|
| Sycophancy Causes | P001 | "LLMs trained with RLHF may learn to agree" | Y | Explains *why* sycophancy happens | RLHF creates misaligned incentives |
| Sycophancy Causes | P003 | "Instruction-following training amplifies agreement" | Y | Different cause, same effect | Both training and data influence behavior |
| Sycophancy Detection | P002 | "Multi-modal models show 67% agreement bias" | Y | Quantifies the problem | Multimodal domain worse than text-only |
| Clinical Application | P005 | "LLMs in medical notes hallucinates 12% of facts" | N | Contradicts P006's "3% error rate" | Different datasets/metrics? Check methods. |
```

**Why this matters for your report:**
- You'll see that sycophancy **causes** include [training, data, architecture]
- You'll spot **disagreements** (P005 vs P006 on error rates) — investigate why
- You'll identify **gaps** ("nobody studied sycophancy in clinical NLP with small LLMs")

### 5.2 Identify Gaps & Your Contribution

Ask:
1. **What has been studied heavily?** (e.g., sycophancy in GPT-4)
2. **What's missing?** (e.g., sycophancy in open-source models, in non-English languages, in real clinical settings)
3. **Where do papers contradict?** (e.g., one says faithfulness improves with scale; another says it doesn't)
4. **Where could a new approach help?** (e.g., a new evaluation metric, a new mitigation technique)

**This gap analysis becomes your motivation in the report Introduction.**

### 5.3 Group Papers by Theme for Your Report Structure

Create a **theme map** (visual or document):

```
REPORT OUTLINE
├── Introduction: The Problem
│   └── Papers: Zhao (2025), Wei (2024) [foundational definitions]
├── Background: Why Sycophancy Matters
│   ├── Causes of sycophancy
│   │   └── Papers: Anthropic (2024), Thenraj (2023)
│   ├── Impact on Faithfulness
│   │   └── Papers: Lee (2025), Camburu (2024)
│   └── Clinical Domain Specificity
│       └── Papers: Cheng (2025), CARE-AD (2025)
├── Methods: How to Detect/Mitigate
│   └── Papers: Pi (2025), Holter (2025), Meinke (2024)
├── Evaluation Approaches
│   └── Papers: Lundberg (2024), BIG-Bench (2023)
└── Open Questions & Gaps
    └── Synthesis of gaps across all themes
```

**Each theme group = one section of your report with 3–5 supporting papers.**

---

## Phase 6: Supporting Your Arguments (Report Writing Phase)

### 6.1 Three Levels of Evidence for Your Report

**Level 1: Foundational Claims** (use 2–3 papers)
> "Sycophancy—the tendency of LLMs to agree with user statements regardless of truthfulness—has been documented in recent work [Wei et al., 2024; Pi et al., 2025]."

**Level 2: Mechanistic Understanding** (use 1 deep paper + background)
> "This phenomenon likely stems from RLHF training objectives that reward agreement as a proxy for helpfulness [Anthropic, 2024]. Recent analysis of multimodal LLMs extends this to visual domains, showing similar bias patterns even when grounded truth contradicts user claims [Pi et al., 2025]."

**Level 3: Domain-Specific Application** (use 1–2 papers directly relevant to your work)
> "In clinical settings, where factual accuracy is safety-critical, sycophancy poses acute risks [Cheng et al., 2025]. Evaluations using LLMs-as-judges to score clinical summaries must account for this bias [Cheng et al., 2025; Lundberg, 2024]."

### 6.2 Making Explicit Connections in Your Writing

**Bad connection (no synthesis):**
> "Wei et al. (2024) studied alignment faking. Pi et al. (2025) studied sycophancy in multimodal LLMs. Thenraj (2023) reduced sycophancy with synthetic data."

**Good connection (synthesis):**
> "Alignment faking [Wei et al., 2024] and sycophancy [Pi et al., 2025] represent related but distinct failure modes: the former involves deliberate deception, while the latter reflects learned agreement bias. Mitigation strategies, such as synthetic data augmentation [Thenraj, 2023], show promise but have not been tested in clinical contexts where the stakes of false agreement are highest."

**Key phrases for synthesis:**
- "Extending the work of X, recent studies show..."
- "In contrast to X's findings, Y demonstrates..."
- "Building on these foundations, X and Y together suggest..."
- "This aligns with but complicates the picture painted by..."

---

## Phase 7: The Full Cycle in 8-12 Weeks (Timeline)

| Week(s) | Task | Output | Time |
|---------|------|--------|------|
| 1–2 | Define research Q + extract keywords | Keyword list (20–30 terms) | 5 hrs |
| 2–3 | Initial database searches (3–5 queries) | 50–100 candidate papers | 8 hrs |
| 3 | First-pass filtering (skim abstracts) | 20–30 papers selected for skim | 6 hrs |
| 3–4 | Skim selected papers + build Literature Matrix | Spreadsheet with 20–30 papers | 10 hrs |
| 4–5 | Citation tracking (identify key papers) | 5–10 additional high-priority papers | 4 hrs |
| 5–7 | Deep read (10–15 key papers) + structured notes | Detailed notes for each core paper | 30 hrs |
| 7–8 | Build Synthesis Matrix | Themes identified + connections mapped | 6 hrs |
| 8–10 | Write first draft of literature review | 2,000–3,000 words with citations | 20 hrs |
| 10–12 | Refine, fact-check, polish | Final literature review | 10 hrs |
| **Total** | | | **~100 hrs** |

**For someone coding in parallel:** Distribute weeks. Do search + skim (Weeks 1–4) while coding prototypes. Shift to deep reading + writing (Weeks 5–8) as code stabilizes.

---

## Phase 8: Common Pitfalls to Avoid

### ❌ Pitfall 1: Reading Every Paper Fully
- **Fix:** Use the Skim-Decide rule. Most papers merit only 15–20 min.

### ❌ Pitfall 2: Disorganized Note-Taking
- **Fix:** Use the structured template. One note file per paper. Sync to your synthesis matrix weekly.

### ❌ Pitfall 3: Missing Recent Preprints
- **Fix:** Check arXiv weekly. Set Google Scholar alerts for your keywords.

### ❌ Pitfall 4: Summarizing Without Synthesis
- **Fix:** Your report should *compare* papers, not list them. "X found A. Y found B." is bad. "X and Y differ on A/B because..." is good.

### ❌ Pitfall 5: Overstating Connections
- **Fix:** If two papers aren't actually related, don't force it. Say so in your report.

### ❌ Pitfall 6: Forgetting to Update Your Matrix
- **Fix:** Every Friday, spend 15 min adding new papers to your spreadsheet and synthesis matrix.

---

## Quick Reference: Decision Tree for Each Paper

```
Found a new paper?
│
├─ Title + Abstract immediately relevant to my research Q?
│  ├─ YES → Check venue. Published in ACL/NeurIPS/ICLR or recent arXiv?
│  │  ├─ YES → FULL READ (45–60 min)
│  │  └─ NO → SKIM (15–20 min)
│  └─ NO → Tangentially related?
│     ├─ YES → Methodological innovation or new dataset?
│     │  ├─ YES → SKIM (15–20 min)
│     │  └─ NO → SKIP
│     └─ NO → SKIP (unless cited by ≥5 other papers in your list)
│
└─ Decision made. Log in Literature Matrix. Move on.
```

---

## Tool Stack Recommendation

**For AI/ML paper literature review:**

| Tool | Purpose | Why |
|------|---------|-----|
| **arXiv + ACL Anthology** | Paper discovery | Fast, indexed by topic, no paywalls |
| **Google Scholar** | Citation tracking + alerts | Tells you who cited your key papers |
| **Zotero or Mendeley** | Reference management | Export BibTeX, organize by folder |
| **Google Sheets** | Literature Matrix + Synthesis Matrix | Filterable, shareable, sortable |
| **Notion or Markdown (Git)** | Detailed paper notes | One file per paper, version control |
| **Overleaf (with BibTeX)** | Writing + citing | Auto-generates citations, integrates with Zotero |

---

## Final Checklist Before Writing Your Report

- [ ] **20–30 papers** in your Literature Matrix
- [ ] **10–15 papers** with detailed notes
- [ ] **Synthesis Matrix** complete with themes identified
- [ ] **Gap analysis** document (what's missing)
- [ ] **Citation file** (BibTeX or equivalent) with all papers
- [ ] **Theme outline** for your report structure
- [ ] **No more than 3 papers** that you've only skimmed, all others read or justified as skim-only
- [ ] **All PDFs organized** in a folder (with file names: `Author_Year_Title.pdf`)
- [ ] **Weekly update log** showing search iterations and decisions

---

**Author's Note:**  
This cycle is based on University of Northampton, Sheffield, and peer institution guidelines, adapted for AI/ML publishing where speed + quality matter. The emphasis on *efficient skimming* and *synthesis* (vs. reading everything) reflects the reality that AI papers accumulate 50+ per day on arXiv alone. Your job is to find the **signal** (1–2% of papers), not read the **noise**. Good luck!
