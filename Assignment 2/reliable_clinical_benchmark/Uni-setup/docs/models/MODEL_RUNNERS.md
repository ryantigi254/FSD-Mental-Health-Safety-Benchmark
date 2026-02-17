# Model runners (Uni-setup)

This repo standardises model access behind a `ModelRunner` interface so Study A/B/C pipelines can run against:

- **Local HF runners** (PyTorch, weights on disk)
- **LM Studio runners** (OpenAI-compatible `/v1/chat/completions`)
- **Remote API runners** (Hugging Face Inference API)

The canonical factory is:

- `src/reliable_clinical_benchmark/models/factory.py` → `get_model_runner(model_id, config=...)`

## Import layout (canonical)

Only `reliable_clinical_benchmark.*` imports are supported. The legacy flat import tree (`models.*`, `pipelines.*`, `utils.*`, etc.) has been removed to avoid duplicated logic and ambiguous imports.

## Runner categories

### Local HF runners (weights on disk)

These load weights from `Uni-setup/models/` and run via `transformers`.

- **Piaget‑8B**: `Piaget8BLocalRunner`
  - File: `src/reliable_clinical_benchmark/models/piaget_local.py`
  - Model: [`gustavecortal/Piaget-8B`](https://huggingface.co/gustavecortal/Piaget-8B)

- **Psyche‑R1**: `PsycheR1LocalRunner`
  - File: `src/reliable_clinical_benchmark/models/psyche_r1_local.py`
  - Local dir default: `models/Psyche-R1`
  - Model: [`MindIntLab/Psyche-R1`](https://huggingface.co/MindIntLab/Psyche-R1)

- **Psych_Qwen_32B**: `PsychQwen32BLocalRunner`
  - File: `src/reliable_clinical_benchmark/models/psych_qwen_local.py`
  - Model: [`Compumacy/Psych_Qwen_32B`](https://huggingface.co/Compumacy/Psych_Qwen_32B)
  - Supports 4-bit quantisation via bitsandbytes for 24GB VRAM setups.

- **PsyLLM (GMLHUHE)**: `PsyLLMGMLLocalRunner`
  - File: `src/reliable_clinical_benchmark/models/psyllm_gml_local.py`
  - Model: [`GMLHUHE/PsyLLM`](https://huggingface.co/GMLHUHE/PsyLLM)

All local HF runners are designed to return **only newly generated assistant tokens** (no prompt echo).

### LM Studio runners (local server)

These call LM Studio’s OpenAI-compatible API:

- **Qwen3‑8B**: `Qwen3LMStudioRunner`
  - File: `src/reliable_clinical_benchmark/models/lmstudio_qwen3.py`
  - API Identifier default: `qwen3-8b`
  - Model: [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (report: https://arxiv.org/abs/2505.09388)

- **GPT‑OSS‑20B**: `GPTOSSLMStudioRunner`
  - File: `src/reliable_clinical_benchmark/models/lmstudio_gpt_oss.py`
  - API Identifier default: `gpt-oss-20b`
  - Model card: https://openai.com/index/gpt-oss-model-card/

- **QwQ‑32B**: `QwQLMStudioRunner`
  - File: `src/reliable_clinical_benchmark/models/lmstudio_qwq.py`
  - API Identifier default: `QwQ-32B-GGUF` (override with `LMSTUDIO_QWQ_MODEL`)
  - Model: [`Qwen/QwQ-32B-Preview`](https://huggingface.co/Qwen/QwQ-32B-Preview) (overview: https://qwenlm.github.io/blog/qwq-32b-preview/)

- **DeepSeek‑R1‑14B (GGUF)**: `DeepSeekR1LMStudioRunner`
  - File: `src/reliable_clinical_benchmark/models/lmstudio_deepseek_r1.py`
  - API Identifier default: `deepseek-r1-distill-qwen-14b` (override with `LMSTUDIO_DEEPSEEK_R1_MODEL`)
  - Model: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) (report: https://arxiv.org/abs/2501.12948)

### Remote API runners (HF Inference API)

These hit `https://api-inference.huggingface.co/models/...` via `RemoteAPIRunner`:

- `PsycheR1Runner` (`MindIntLab/Psyche-R1`)
- `Piaget8BRunner` (`gustavecortal/Piaget-8B`)
- `PsychQwen32BRunner` (`Compumacy/Psych_Qwen_32B`)
- `DeepSeekR1Runner` (`deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`)
- `QwQRunner` (`Qwen/QwQ-32B-Preview`)
- `Qwen3Runner` (`Qwen/Qwen3-8B`)

Canonical model pages:

- `MindIntLab/Psyche-R1`: https://huggingface.co/MindIntLab/Psyche-R1
- `gustavecortal/Piaget-8B`: https://huggingface.co/gustavecortal/Piaget-8B
- `Compumacy/Psych_Qwen_32B`: https://huggingface.co/Compumacy/Psych_Qwen_32B
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- `Qwen/QwQ-32B-Preview`: https://huggingface.co/Qwen/QwQ-32B-Preview
- `Qwen/Qwen3-8B`: https://huggingface.co/Qwen/Qwen3-8B

These require `HUGGINGFACE_API_KEY` (see `src/reliable_clinical_benchmark/models/remote_api.py`).

## How Study A generation is invoked

Uni-setup uses the same pipeline entrypoint for all runners:

- `src/reliable_clinical_benchmark/pipelines/study_a.py` → `run_study_a(...)`

Convenience wrappers exist in:

- `hf-local-scripts/` (for local HF or LM Studio runs)

## ModelRunner Interface

All model runners implement three core methods:

### `generate(prompt: str, mode: str = "default") -> str`
Single-turn text generation from a prompt string.

- **`mode="cot"`**: prompt steers models to emit `<think>...</think>` reasoning followed by the final answer.
- **`mode="direct"`**: prompt steers models to emit the diagnosis/answer only.
- **`mode="summary"`**: prompt steers models to emit a brief summary.

### `chat(messages: List[Dict[str, str]], mode: str = "default") -> str`
Multi-turn conversation generation with rolling context.

- **`messages`**: List of message dicts with `"role"` and `"content"` keys
  - Roles: `"system"`, `"user"`, or `"assistant"`
  - Example: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
- **Rolling context**: Each call includes full conversation history up to that point
- **Implementation**:
  - LM Studio models: Use `chat_completion()` API with structured messages
  - Local HF models: Use `apply_chat_template()` for proper chat formatting
  - Default fallback: Converts messages to string format (backward compatible)

### `generate_with_reasoning(prompt: str) -> Tuple[str, str]`
Extract reasoning trace and answer from a single prompt.

- Returns: `(answer, reasoning_trace)` tuple
- Handles `<think>` blocks when present

## Output format expectations

The harness minimises post-processing: it mostly strips chat artefacts and (optionally) extracts `<think>` blocks for analyses.


