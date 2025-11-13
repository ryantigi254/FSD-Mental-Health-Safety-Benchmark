# Reliable Clinical Reasoning in LLMs (CSY3055 AS1)

This repository contains my Assignment 1 proposal and supporting coursework for the “Natural Language Processing (CSY3055)” module. The project evaluates reliable clinical reasoning in LLMs across three focused studies: faithfulness, sycophancy pressure, and longitudinal continuity.

Status
- Proposal: see `Assignment/AS1_proposal.tex` (LaTeX source)
- Coursework notes/notebooks: `Wk3`–`Wk6` (module materials kept for context)
- Code: to be added in Assignment 2 phase (re-using the same repository)

How to build the proposal (local)
1. Install LaTeX (e.g., TeX Live or MacTeX).
2. From the repo root:
   ```
   pdflatex Assignment/AS1_proposal.tex
   ```
   or open on Overleaf and compile there.

Repository layout (branch: `assignment`)
- `Assignment/AS1_proposal.tex` – proposal source (Harvard-style references in-text; link-outs in References)
- `Wk3`–`Wk6` – module coursework for reference
- `README.md`, `LICENSE`, `.gitignore`

Key references (links)
- PsyLLM paper (Hu et al., 2025): https://arxiv.org/pdf/2505.15715
- OpenR1‑Psy dataset: https://huggingface.co/datasets/GMLHUHE/OpenR1-Psy
- PsyLLM model card: https://huggingface.co/GMLHUHE/PsyLLM
- Qwen3‑8B model card: https://huggingface.co/Qwen/Qwen3-8B
- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Chain‑of‑Thought: https://arxiv.org/abs/2201.11903
- Self‑Consistency: https://arxiv.org/abs/2203.11171

Notes for reviewers
- This branch preserves history and includes coursework folders for context; sensitive/sample artefacts are ignored where required.
- Commits are GPG‑signed locally (key: `4D8FDE4B5A2A41BA`); historical rewrites (if any) may include unsigned rewritten commits, which is expected.

License
- MIT (see `LICENSE`)
