# Robert Prototype Workspace

This top-level folder carries the initial working prototype for the “Robert” mental-health support chatbot. It is intentionally minimal, favouring a thin slice that the clinical evaluators can exercise quickly while we iterate toward the full reference architecture captured in `Robert Chatbot Prototype.png`.

![Robert System Reference](Robert%20Chatbot%20Prototype.png)

## Repository Layout

| Path | Purpose |
| --- | --- |
| `robert-backend/` | FastAPI service scaffold (health/chat/escalate endpoints, LM Studio / Gemini connectors, safety regex, pytest suite). |
| `robert-frontend/` | Next.js 15 + Tailwind + shadcn UI (safety banner, distress scale, quick actions, memory consent, Netlify build config, vitest + jest-axe). |
| `models/` | Local model snapshots (e.g., `models/PsyLLM/` downloaded from Hugging Face for on-device inference). |
| `Robert Chatbot Prototype.png` | Full system blueprint agreed with supervisors/clinicians. The live code is a thin slice of this diagram. |

Each inner directory contains its own README with setup steps (`robert-backend/README.md`, `robert-frontend/README.md`).

### Local PsyLLM + MLX toolchain

We keep PyTorch (`models/PsyLLM/`) and MLX (`models/PsyLLM-mlx/`) weights side-by-side. To generate an MLX export:

```bash
conda create -y -n mlx-env python=3.11
conda run -n mlx-env python -m pip install mlx mlx-lm --break-system-packages
conda run -n mlx-env python scripts/convert_psyllm_to_mlx.py --force --output ./models/PsyLLM-mlx -q
```

The conversion uses `mlx_lm.convert` ([docs](https://github.com/ml-explore/mlx-lm#command-line)), and Apple’s MLX runtime can comfortably host the 7B PsyLLM model on the M3 Max (48 GB RAM) — see the MLX “Large Models” guidance for wired-memory tuning if you approach the RAM ceiling ([MLX README](https://github.com/ml-explore/mlx-lm#large-models)). Run the FastAPI service under this environment (`conda run -n mlx-env uvicorn ...`) to take advantage of the MLX backend.

## Iterative Delivery Plan

Although the reference diagram depicts a multi-agent planner/verifier/executor loop, we are building it incrementally with clinician feedback after each milestone:

1. **Thin slice (current)**  
   - UI safety shell: banner + quick exit + crisis copy.  
   - Distress slider, memory consent, minimal chat loop.  
   - FastAPI orchestrator with a single provider (LM Studio/Gemini) and regex-based crisis interception.

2. **Signals & audit (next)**  
   - Integrate `sentinet/suicidality` + `SamLowe/roberta-base-go_emotions`.  
   - Persist plan/resource selections to structured logs.  
   - Enable local specialist models (e.g., PsyLLM) via `models/`.

3. **Planner / verifier introduction**  
   - Wrap Psyche-R1 as planner + verifier producing JSON `{action, boundary_flags, must_fix[]}`.  
   - DBT skill selector feeding the UI’s quick actions.  
   - Supervisor gate enforcing `approve`/`fix`/`fallback`/`crisis`.

4. **Full compliance and tooling**  
   - Expand the audit store (SQLite + encryption) as shown in the diagram.  
   - Optional tool integrations (psychoeducation fact cards, external resource lookups).  
   - Harden incident reporting, observability, and evaluator dashboards.

At every stage we pause for clinician sign-off before moving to the next layer. This keeps the build aligned with the evaluation cadence and ensures we can halt or adjust easily if safety reviewers flag gaps.

## Working With Local Models

The `models/` directory is intentionally separate from the backend/frontend code so large checkpoints (e.g., PsyLLM) do not leak into git history. Use `huggingface_hub.snapshot_download` to populate it, then point the backend provider at the local path.

## Tracking Progress

- Backend test suite: `cd robert-backend && source .venv/bin/activate && pytest`.  
- Frontend tests: `cd robert-frontend && npm run test && npm run test:a11y`.  
- Production dry-run: `cd robert-frontend && npm run build && npm run start`.

Future enhancements should update this README and the diagram if the architecture changes materially.

