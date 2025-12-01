# Setup

## Structure

- `psy-llm-local/` - Direct PyTorch inference script for PsyLLM
- `lmstudio-scripts/` - Scripts for querying models via LM Studio API

## PsyLLM Inference

1. Install dependencies:
```bash
pip install torch transformers accelerate
```

2. Download the PsyLLM model:
```bash
cd psy-llm-local/models/PsyLLM
huggingface-cli login  # if needed
huggingface-cli download GMLHUHE/PsyLLM --local-dir . --local-dir-use-symlinks False
```

3. Run inference:
```bash
cd psy-llm-local
python infer.py
```

## LM Studio Scripts

1. Install dependencies:
```bash
pip install requests
```

2. Start LM Studio server with your model loaded (default: http://127.0.0.1:1234)

3. Run a script:
```bash
cd lmstudio-scripts
python chat_gpt_oss_20b.py "Your prompt here"
# or
python chat_qwen3_8b.py "Your prompt here"
```

## Model IDs

- OpenAI GPT-OSS-20B: `openai_gpt-oss-20b`
- Qwen3-8B: `qwen3-8b`

Adjust the `MODEL_ID` in the scripts if your LM Studio model identifier differs.

