import os
import torch

from config import (
    batch_size,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    max_iters,
    eval_interval,
    eval_iters,
    learning_rate,
    device,
    data_path,
    checkpoint_path,
)

from dataset import CharTokenizer, NextCharacterDataset, load_text_data, split_train_val
from model import TinyGPT


def main():
    print("GPT2 project check start")
    print("-" * 60)

    print("[1] Config check")
    print(f"batch_size: {batch_size}")
    print(f"block_size: {block_size}")
    print(f"n_embd: {n_embd}")
    print(f"n_head: {n_head}")
    print(f"n_layer: {n_layer}")
    print(f"dropout: {dropout}")
    print(f"max_iters: {max_iters}")
    print(f"eval_interval: {eval_interval}")
    print(f"eval_iters: {eval_iters}")
    print(f"learning_rate: {learning_rate}")
    print(f"device: {device}")
    print(f"data_path: {data_path}")
    print(f"checkpoint_path: {checkpoint_path}")

    if n_embd % n_head != 0:
        raise ValueError("n_embd must be divisible by n_head")

    print("\n[2] Dataset check")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    text = load_text_data(data_path)
    print(f"number of characters: {len(text):,}")

    tokenizer = CharTokenizer(text)
    encoded = tokenizer.encode(text)

    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"encoded length: {len(encoded):,}")

    train_data, val_data = split_train_val(encoded, train_ratio=0.9)
    train_dataset = NextCharacterDataset(train_data, block_size)
    val_dataset = NextCharacterDataset(val_data, block_size)

    print(f"train dataset length: {len(train_dataset):,}")
    print(f"validation dataset length: {len(val_dataset):,}")

    x, y = train_dataset[0]
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    print("\n[3] Model forward check")

    model = TinyGPT(tokenizer.vocab_size).to(device)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"number of parameters: {num_params:,}")

    x_batch = x.unsqueeze(0).to(device)
    y_batch = y.unsqueeze(0).to(device)

    logits, loss = model(x_batch, y_batch)

    print(f"logits shape: {logits.shape}")
    print(f"loss: {loss.item():.4f}")

    print("\n[4] Generate check")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=50)

    generated_text = tokenizer.decode(generated[0].tolist())

    print("generated sample:")
    print(generated_text)

    print("\nGPT2 project check completed successfully.")


if __name__ == "__main__":
    main()