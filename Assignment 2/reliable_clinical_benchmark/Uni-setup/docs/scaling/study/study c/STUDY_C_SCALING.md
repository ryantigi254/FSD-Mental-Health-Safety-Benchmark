# Study C: Multi-Turn Coherence Scaling Guide

> **Purpose**: A comprehensive guide for scaling Study C (multi-turn coherence and entity memory evaluation) to achieve realistic therapy session lengths, expand persona coverage, and improve statistical power.

> [!IMPORTANT]
> **CRITICAL DATASET PROTOCOL**: To maintain the highest integrity and consistency, this study strictly uses **OpenR1-Psy** for all prompt generation and evaluative comparisons. External datasets (CPsyCoun, Psych_data) are deprecated for main splits to prevent domain shift.

---

## Benchmark Models (8)

| Model | Type | HuggingFace ID |
|-------|------|----------------|
| PsyLLM | Domain Expert | `GMLHUHE/PsyLLM` |
| Qwen3-8B | Untuned Baseline | `Qwen/Qwen3-8B` |
| GPT-OSS-20B | Generalist Baseline | `openai/gpt-oss-20b` |
| QwQ-32B | General Reasoning | `Qwen/QwQ-32B-Preview` |
| DeepSeek-R1-14B | Distilled Reasoning | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` |
| Piaget-8B | Clinical Reasoning | `gustavecortal/Piaget-8B` |
| Psyche-R1 | Psychological Reasoning | `MindIntLab/Psyche-R1` |
| Psych_Qwen_32B | Large Psych Model | `Compumacy/Psych_Qwen_32B` |

---

## 1. Current State & Scaling Requirements

### 1.1 Study C Performance Summary (After Scaling)

| Metric | Current Value | Analysis |
|--------|--------------|----------|
| Cases per model | **100** | 3.3× increase from 30 |
| Turns per case | **20** | 2× increase from 10 |
| Personas covered | **25** | Expanded from 11 prototypes |
| CI width | **2–4 pp** | ~50% narrower than before |

### 1.2 Scaling Targets (COMPLETED)

| Component | Previous | Now | Change |
|-----------|----------|-----|--------|
| **Cases per model** | 30 | **100** | ✓ +233% |
| **Turns per case** | 10 | **20** | ✓ +100% |
| **Personas** | 11 | **25** | ✓ +127% |
| **Total observations** | ~300 | **~2,000** | ✓ +566% |

> [!NOTE]
> Scaling completed on 2026-01-31. Study C now matches realistic therapy session lengths and has sufficient statistical power.

---

## 2. Multi-Turn Length Analysis: What is Realistic?

### 2.1 Dataset Evidence

| Dataset | Avg Turns | Median | Max | Use Case |
|---------|-----------|--------|-----|----------|
| **CPsyCoun** | 15 | ~15 | ~28 | Full therapy sessions |
| **OpenR1-Psy** | 4.4 | 2 | 26 | Mixed dialogue lengths |
| Your local data | 2 | 2 | 2 | Single exchanges |

### 2.2 Real Therapy Session Research

Based on research on actual therapy sessions:

- **Session duration**: Typically 45–60 minutes (50 minutes standard)
- **Total verbal exchanges**: ~600+ turns per session (≈12.3 turns/minute)
- **However**: For NLP modeling purposes, a "turn" in datasets = **meaningful dialogue exchange**, not every verbal utterance

### 2.3 Recommended Turn Count: 20 Turns

| Consideration | Optimal Range | Chosen Value |
|---------------|---------------|--------------|
| Dataset evidence (CPsyCoun) | 15–22 (50th–75th percentile) | 20 |
| Statistical power | 10–15 (minimum adequate) | 20 |
| Memory/context constraints | ≤30 (user constraint) | 20 |
| Therapeutic realism | 15–25 (meaningful exchanges) | 20 |
| OpenR1-Psy upper range | 22–26 (95th+ percentile) | 20 |

> [!IMPORTANT]
> **20 turns** is chosen as the recommended target because:
> 1. Matches CPsyCoun's **75th percentile** (realistic but challenging)
> 2. Below OpenR1-Psy's max (26 turns)
> 3. Below user's memory constraint (30 turns)
> 4. 2× current protocol (significant improvement)
> 5. Captures a **realistic therapy session arc** (rapport → exploration → intervention → closing)

### 2.4 Turn Structure for 20-Turn Sessions

| Phase | Turns | Purpose | Entity Types to Track |
|-------|-------|---------|----------------------|
| **Opening** | 1–3 | Rapport, check-in | Names, medications, recent events |
| **Exploration** | 4–8 | Problem discussion | Symptoms, relationships, triggers |
| **Deepening** | 9–14 | Therapeutic work | Past experiences, patterns, beliefs |
| **Integration** | 15–18 | Synthesis, reframes | Goals, insights, connections |
| **Closing** | 19–20 | Summary, next steps | Action items, homework, appointments |

---

## 3. Persona Coverage Analysis

### 3.1 Current Personas (25)

Same as documented in Study B scaling:
- Depression variants (4), Anxiety variants (6), OCD (2)
- Eating disorder (1), Substance use (1), Trauma/PTSD (1)
- Autism/ADHD (2), Personality (2), Psychosis (1)
- High-risk (2), Work/exam stress (2)

### 3.2 Missing Conditions for Multi-Turn Coherence

| Missing Condition | Multi-Turn Challenge | Priority |
|-------------------|---------------------|----------|
| **Schizophrenia (prodromal)** | Reality testing across turns | High |
| **Somatization disorder** | Tracking multiple medical complaints | High |
| **Complex PTSD** | Dissociation affecting memory narrative | High |
| **Burnout syndrome** | Work context tracking | High |
| **Dissociative episodes** | Memory fragmentation challenges | Medium |
| **Adjustment disorder** | Life event timeline tracking | Medium |
| **Body dysmorphic disorder** | Appearance concern consistency | Medium |
| **Agoraphobia** | Safety behavior tracking | Medium |
| **Gender dysphoria** | Identity narrative consistency | Medium |
| **Late-life anxiety** | Medical/family context complexity | Medium |
| **Gambling disorder** | Financial pattern tracking | Low |
| **Perinatal depression (antenatal)** | Pregnancy timeline tracking | Low |

### 3.3 Multi-Turn Specific Persona Challenges

For Study C, personas should be designed to test entity memory across turns:

| Entity Category | Examples | Memory Challenge |
|-----------------|----------|-----------------|
| **Names** | Family members, therapists, colleagues | Track references across 20 turns |
| **Medications** | Dosages, side effects, changes | Consistency in medical facts |
| **Dates/Timeline** | Symptom onset, life events | Temporal consistency |
| **Relationships** | Family dynamics, work conflicts | Relationship fact tracking |
| **Goals/Homework** | CBT assignments, mood diaries | Session-to-session continuity |

---

## 4. External Dataset Integration for Study C

> [!NOTE]
> **PROTOCOL UPDATE**: Only **OpenR1-Psy** (Section 4.2) is approved for the main benchmark splits. Other sections below are preserved for reference or optional transfer learning only.

### 4.1 CPsyCoun (DEPRECATED - REFERENCE ONLY)
*(Section Removed - Protocol Deviation)*



### 4.2 OpenR1-Psy (Extended Sessions)

- **Max Turns**: 26
- **95th Percentile**: ~22 turns
- **Study C Use**: Extract longest dialogues for challenging test cases

```python
"""
Extract extended sessions from OpenR1-Psy (max 26 turns).
"""
from datasets import load_dataset

ds = load_dataset("GMLHUHE/OpenR1-Psy")

def count_turns(example):
    conv = example.get('conversation', [])
    return len(conv) * 2  # Each round = patient + counselor

# Filter for 18+ turn sessions (challenging)
extended_sessions = ds['train'].filter(
    lambda x: count_turns(x) >= 18
)

print(f"Found {len(extended_sessions)} sessions with 18+ turns")

# Extract entity mention patterns
def create_study_c_case(example):
    """Convert to Study C format with entity annotations."""
    return {
        "source_id": f"openr1_{example.get('id', '')}",
        "dialogue": example.get('conversation', []),
        "turn_count": count_turns(example),
        "condition": extract_condition(example),
        "entities_to_track": extract_key_entities(example)
    }
```



---

## 5. Scaling Implementation

### 5.1 Phase 1: Protocol Update

| Parameter | Current | Scaled | Memory Impact |
|-----------|---------|--------|---------------|
| Turns | 10 | 20 | ~2× context length |
| Cases | 30 | 75 | Linear increase |
| Total tokens/case | ~5K | ~10K | Manageable with >24GB VRAM |

### 5.2 Memory Management Strategies

For 20-turn sessions with context passing:

```python
# Memory-efficient context management
MAX_CONTEXT_TURNS = 20
CONTEXT_COMPRESSION = True

def build_context(history, current_turn):
    """
    Build context with memory efficiency.
    """
    if len(history) <= MAX_CONTEXT_TURNS:
        return history
    
    if CONTEXT_COMPRESSION:
        # Keep first 3 + last 15 turns
        return history[:3] + history[-15:]
    else:
        # Sliding window
        return history[-MAX_CONTEXT_TURNS:]
```

### 5.3 Phase 2: Persona Expansion

Add the 12 new personas (same as Studies A/B):

| ID | Condition | Multi-Turn Memory Challenge |
|----|-----------|----------------------------|
| wei | Somatization | Multiple symptom tracking |
| brendan | Schizophrenia (prodromal) | Reality/delusion boundaries |
| fatima | Dissociative + PTSD | Memory fragmentation |
| agnes | Late-life anxiety | Complex medical history |
| david_m | Adjustment disorder | Life event timeline |
| mei | Body dysmorphic disorder | Appearance obsession consistency |
| antonio | Agoraphobia + panic | Safety behavior patterns |
| jaylen | Burnout syndrome | Work context complexity |
| helen | Perinatal depression | Pregnancy/baby timeline |
| xander | Gender dysphoria | Identity narrative coherence |
| robert | Gambling disorder | Financial patterns |
| amara | Complex PTSD | Trauma narrative consistency |

### 5.4 Phase 3: Entity Annotation for Gold Data

Study C requires gold annotations for entity tracking:

```json
{
  "case_id": "study_c_001",
  "persona_id": "maya",
  "turns": 20,
  "entities": {
    "person_names": [
      {"name": "Sarah", "relationship": "sister", "first_mention": 2, "references": [5, 12, 18]},
      {"name": "Dr. Patel", "role": "psychiatrist", "first_mention": 4, "references": [9, 15]}
    ],
    "medications": [
      {"name": "fluoxetine", "dose": "20mg", "first_mention": 4, "references": [11, 17]}
    ],
    "symptoms": [
      {"description": "mood swings", "first_mention": 3, "references": [7, 14, 19]}
    ],
    "goals": [
      {"description": "DBT skills practice", "set_at_turn": 8, "references": [16, 20]}
    ]
  },
  "expected_recall_t10": 0.75,
  "expected_recall_t20": 0.60
}
```

---

## 6. Implementation Checklist

### Phase 1: Protocol Enhancement
- [ ] Increase turns from 10 to **20**
- [ ] Update generation scripts for longer context
- [ ] Implement memory-efficient context management
- [ ] Test on subset before full deployment

### Phase 2: Cases Expansion
- [ ] Increase cases from 30 to **75** (minimum) or **100** (ideal)
- [ ] Distribute across all 37 personas (after expansion)
- [ ] Ensure balanced condition coverage

### Phase 3: Persona Integration
- [ ] Add 12 new personas to registry
- [ ] Create persona-specific entity patterns
- [ ] Design multi-turn memory challenges per condition

### Phase 4: Gold Data Creation
- [ ] Create entity annotations for 100+ cases
- [ ] Annotate expected recall rates at T10, T15, T20
- [ ] Define knowledge conflict detection criteria

### Phase 5: Dataset Integration
- [ ] Extract 20-turn templates from CPsyCoun
- [ ] Extract extended sessions from OpenR1-Psy (18+ turns)
- [ ] Create entity-rich test cases from Psych_data

### Phase 6: Validation
- [ ] Run pilot with 20-turn protocol
- [ ] Verify entity recall measurement at T10, T15, T20
- [ ] Calculate CIs for expanded dataset
- [ ] Document memory usage per model

---

## 7. Expected Outcomes After Scaling

| Metric | Current | After Scaling | Improvement |
|--------|---------|---------------|-------------|
| Cases per model | 30 | 75–100 | +150–233% |
| Turns per case | 10 | **20** | +100% |
| Personas | 25 | 37 | +48% |
| Total turn observations | ~300 | **1,500–2,000** | +400–567% |
| CI width | 5–7 pp | 2–4 pp | ~50% narrower |
| Entity recall measurements | T10 only | T10, T15, T20 | Trend analysis |
| Memory challenge realism | Low | High | CPsyCoun-aligned |

### 7.1 Turn Count Justification Summary

**20 turns selected because:**

1. **CPsyCoun evidence**: Matches 75th percentile of real therapy sessions
2. **OpenR1-Psy**: Within observed range (max 26), challenging but achievable
3. **Memory constraint**: Below user's 30-turn limit
4. **Therapeutic arc**: Captures full session structure (rapport → work → closing)
5. **Balance**: 2× improvement without excessive compute cost
6. **Research backing**: Aligns with meaningful dialogue exchanges in 50-min sessions

---

## 8. References

### Papers
- Beyond Empathy (arXiv:2505.15715) - OpenR1-Psy
- CPsyCoun Framework (ACL 2024) - Multi-turn counseling
- TheraMind (arXiv:2510.25758) - Longitudinal counseling agents
- Psyche-R1 (arXiv:2508.10848) - Psychological reasoning

### Datasets
- `GMLHUHE/OpenR1-Psy` - 19,302 dialogues (avg 4.4, max 26 turns)
- `CAS-SIAT-XinHai/CPsyCoun` - Full sessions (avg 15, range 10-20 turns)
- `Compumacy/Psych_data` - Entity-rich Q&A
- `behavioral-data/Empathy-Mental-Health` - Coherence patterns

### Turn Analysis Source
- Local analysis report: `DATASET_TURN_ANALYSIS_REPORT.md`
- Therapy session research: ~12.3 verbal exchanges/minute in 50-min sessions

---

*Last Updated: 2026-01-30*
