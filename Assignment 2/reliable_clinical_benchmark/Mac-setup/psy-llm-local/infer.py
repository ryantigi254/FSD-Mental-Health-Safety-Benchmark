import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_DIR = Path(__file__).parent / "models" / "PsyLLM"


def select_device() -> tuple[str, torch.dtype]:
    """
    Select device and dtype for PsyLLM inference.

    Priority:
    1. Honour PSY_DEVICE env var if set (cuda | mps | cpu | mlx)
    2. Otherwise auto-detect: CUDA > MPS > CPU
    """
    requested = os.environ.get("PSY_DEVICE", "").strip().lower()

    # Optional: detect MLX availability (requires separate implementation)
    try:
        import mlx.core  # type: ignore

        mlx_available = True
    except ImportError:
        mlx_available = False

    if requested:
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda", torch.float16
            else:
                print("PSY_DEVICE=cuda requested but CUDA is not available; falling back to auto-detection.")
        elif requested == "mps":
            if torch.backends.mps.is_available():
                # MPS supports float32 robustly; float16 can be flaky on some stacks
                return "mps", torch.float32
            else:
                print("PSY_DEVICE=mps requested but MPS is not available; falling back to auto-detection.")
        elif requested == "cpu":
            return "cpu", torch.float32
        elif requested == "mlx":
            print(
                "PSY_DEVICE=mlx requested, but this script uses PyTorch/transformers.\n"
                "Use the MLX conversion flow described in Assignment 2/docs/psyllm_setup.md. "
                "Falling back to auto device selection for this script."
            )
        else:
            print(f"Unknown PSY_DEVICE='{requested}', falling back to auto-detection.")

    # Auto-detect path (no or unusable PSY_DEVICE)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    if mlx_available:
        print(
            "Note: MLX is installed but this script is running the PyTorch path. "
            "For MLX, use the converted weights and sandbox provider as per psyllm_setup.md."
        )

    return device, dtype


def main() -> None:
    device, dtype = select_device()

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"Loading model on {device} with dtype {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype)
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


