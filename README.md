# Reliable Clinical Reasoning in LLMs (CSY3055 AS1)

This repository hosts my Assignment work for the CSY3055 Natural Language Processing module.  
The project proposes a lightweight benchmark to evaluate **clinical reasoning reliability without retrieval** in large language models destined for mental‑health support tooling.

## Alignment safety focus

Our benchmark evaluates three failure modes relevant to alignment safety: **unfaithful reasoning** (Study A), which can reveal when models 'scheme' by producing correct answers with fabricated rationales; **sycophantic agreement** (Study B), where models strategically prioritize user approval over truth; and **longitudinal drift** (Study C), which may indicate inconsistent strategic behavior across sessions. These failure modes are particularly critical in mental‑health applications, where models must maintain both clinical accuracy and safety boundaries even under pressure.

## Project snapshot
- **AS1 deliverable:** `Project Proposal/NLP Project Proposal by Ryan Gichuru.pdf` – the proposal describing motivation, literature, methodology, ethics, and project plan.
- **AS2 deliverable:**
  - `data/`: OpenR1–Psy IDs, empathy/pressure prompts, multi–turn scripts, labelling guide, `policy.md`
  - `src/`: one runner for all three studies, metrics with bootstrap CIs, plotting, configuration files
  - `runs/`: raw generations and per–slice CSVs
  - `reports/`: 4–6 page PDF with headline figures and failure examples
  - `README.md`: one–command reproduce, headline table, limitations, licence
- **Models under test:** PsyLLM (domain expert), Qwen/Qwen3‑8B (untuned baseline) and openai/gpt‑oss‑20b (larger open reasoning baseline).

## Planned studies & metrics
1. **Study A – Faithfulness on OpenR1‑Psy**  
   - *Question:* Do step‑by‑step rationales line up with the gold traces?  
   - *Signals:* Step‑F1, Final Accuracy, Faithfulness Gap (correct answer but unfaithful reasoning).  
   - *Interventions:* Direct answer vs Chain‑of‑Thought vs Self‑Critique.
2. **Study B – Empathy vs Truth under Social Pressure**  
   - *Question:* Can we keep empathy while refusing to parrot user errors?  
   - *Signals:* AgreementRate, Accuracy, Truth‑Under‑Pressure, effect of an "empathy‑then‑correct" scaffold.
3. **Study C – Longitudinal Therapeutic Continuity**  
   - *Question:* Does a low‑cost session memory reduce drift over 3‑5 turns?  
   - *Signals:* Continuity Score (MiniLM embeddings + cosine), Safety Drift Rate, Refusal/Redirect Rate.

## Repository layout
| Path | Description |
| --- | --- |
| `Project Proposal/NLP Project Proposal by Ryan Gichuru.pdf` | **AS1 deliverable** – the submitted proposal (PDF). |
| `Assignment/AS1_proposal.tex` | LaTeX source for the proposal (for reference/editing). |
| `Assignment/Misc/` | Reference papers grouped by topic. |
| `Assignment/Samples/` | Marking exemplars for scope calibration. |
| `Wk1`, `Wk2`, `Wk 3`, `Course Material/` | Module notes and coursework (kept for transparency). |
| `LICENSE` | MIT licence. |

## How to build the proposal (local)
1. Install LaTeX (e.g., TeX Live or MacTeX).
2. From the repo root:
   ```bash
   pdflatex Assignment/AS1_proposal.tex
   ```
   or open on Overleaf and compile there.

## Key references (links)
- PsyLLM paper (Hu et al., 2025): https://arxiv.org/pdf/2505.15715
- OpenR1‑Psy dataset: https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy
- PsyLLM model card: https://huggingface.co/GMLHUHE/PsyLLM
- Qwen3‑8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Chain‑of‑Thought: https://arxiv.org/abs/2201.11903
- Self‑Consistency: https://arxiv.org/abs/2203.11171

## Next steps
1. Finalise prompt templates, policy doc, and Step‑F1 matcher skeleton (Week 1‑2 plan).
2. Implement the shared runner + metrics (Weeks 2‑3).
3. Publish the small benchmark and accompanying write‑up in the same repository for Assignment 2 / dissertation alignment.

## Notes for reviewers
- This branch preserves history and includes coursework folders for context; sensitive/sample artefacts are ignored where required.
- Commits are GPG‑signed locally (key: `4D8FDE4B5A2A41BA`); historical rewrites (if any) may include unsigned rewritten commits, which is expected.

## License
- MIT (see `LICENSE`)

Questions or feedback? Open an issue or ping me at `ryantigi2020@gmail.com`.
