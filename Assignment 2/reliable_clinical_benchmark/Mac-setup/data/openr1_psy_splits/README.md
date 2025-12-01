## OpenR1-Psy frozen splits (Study A/B/C)

This directory holds the **frozen JSON test splits** used by the benchmark pipelines:

- `study_a_test.json` – faithfulness vignettes with gold reasoning
- `study_b_test.json` – sycophancy prompts (control + injected) and optional multi-turn cases
- `study_c_test.json` – longitudinal multi-turn cases for drift

The code expects these files exactly here (paths are relative to the Mac-setup root):

- Study A loader: `data/openr1_psy_splits/study_a_test.json`
- Study B loader: `data/openr1_psy_splits/study_b_test.json`
- Study C loader: `data/openr1_psy_splits/study_c_test.json`

### JSON schemas

**Study A (`study_a_test.json`)**

```json
{
  "samples": [
    {
      "id": "a_001",
      "prompt": "<clinical vignette text>",
      "gold_answer": "<gold diagnosis label>",
      "gold_reasoning": [
        "step 1 ...",
        "step 2 ..."
      ]
    }
  ]
}
```

**Study B (`study_b_test.json`)**

Single‑turn sycophancy items:

```json
{
  "samples": [
    {
      "id": "b_001",
      "prompt": "<neutral clinical question/vignette>",
      "gold_answer": "<gold diagnosis label>",
      "incorrect_opinion": "<user's incorrect bias string>",
      "metadata": {}
    }
  ],
  "multi_turn_cases": [
    {
      "id": "b_mt_001",
      "gold_answer": "<gold diagnosis label>",
      "turns": [
        { "turn": 1, "message": "<user message with mild pressure>" },
        { "turn": 2, "message": "<user message with stronger pressure>" }
      ]
    }
  ]
}
```

**Study C (`study_c_test.json`)**

```json
{
  "cases": [
    {
      "id": "c_001",
      "patient_summary": "<initial summary (Turn 1 context)>",
      "critical_entities": [
        "major depressive disorder",
        "fluoxetine 20mg",
        "penicillin allergy"
      ],
      "turns": [
        { "turn": 1, "message": "<patient utterance turn 1>" },
        { "turn": 2, "message": "<patient utterance turn 2>" }
      ],
      "metadata": {}
    }
  ]
}
```

### How to build from OpenR1-Psy (outline)

For Study A you should **derive `study_a_test.json` from the OpenR1‑Psy dataset** (gold reasoning traces), e.g.:

```python
from datasets import load_dataset
import json

ds = load_dataset("GMLHUHE/OpenR1-Psy", split="test")

samples = []
for i, row in enumerate(ds):
    samples.append(
        {
            "id": f"a_{i:03d}",
            "prompt": row["prompt"],
            "gold_answer": row["label"],
            "gold_reasoning": row.get("rationale", []),
        }
    )

with open("data/openr1_psy_splits/study_a_test.json", "w") as f:
    json.dump({"samples": samples}, f, indent=2)
```

Study B and C prompts (sycophancy opinion injections and longitudinal scripts) are **synthetic** and should be authored once, then frozen into `study_b_test.json` and `study_c_test.json` following the schemas above.


