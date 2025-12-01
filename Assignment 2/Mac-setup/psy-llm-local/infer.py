import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models" / "PsyLLM"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"Loading model on {device} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    if device == "cuda":
        model = model.to(device)

    prompt = (
        "<|im_start|>system\n"
        "You are PsyLLM, a compassionate therapeutic assistant. Provide concise, empathetic support."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "I've been feeling burnt out at work and need some ideas to cope."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("Generating...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    completion = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print("\n===== RAW OUTPUT =====\n")
    print(completion)

    assistant_reply = completion.split("<|im_start|>assistant")[-1]
    assistant_reply = assistant_reply.replace("<|im_end|>", "").strip()

    print("\n===== CLEANED ASSISTANT REPLY =====\n")
    print(assistant_reply)


if __name__ == "__main__":
    main()

