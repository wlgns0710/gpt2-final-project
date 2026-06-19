import torch

from config import checkpoint_path, device, max_new_tokens
from dataset import CharTokenizer
from model import TinyGPT


def main():
    print("GPT 2.0 text generation start")
    print(f"device: {device}")

    # ---------------------------------------------------------
    # 1. Load checkpoint
    # ---------------------------------------------------------
    checkpoint = torch.load(checkpoint_path, map_location=device)

    chars = checkpoint["chars"]
    vocab_size = checkpoint["vocab_size"]

    # tokenizer 복원
    tokenizer = CharTokenizer("".join(chars))

    print("\n[1] Checkpoint loaded")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"vocab size: {vocab_size}")

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------
    model = TinyGPT(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n[2] Model loaded")

    # ---------------------------------------------------------
    # 3. Generate text
    # ---------------------------------------------------------
    start_text = "The movie"
    start_ids = tokenizer.encode(start_text)

    context = torch.tensor(
        [start_ids],
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
    )[0].tolist()

    generated_text = tokenizer.decode(generated_ids)

    print("\n[3] Generated text")
    print("-" * 60)
    print(generated_text)
    print("-" * 60)


if __name__ == "__main__":
    main()