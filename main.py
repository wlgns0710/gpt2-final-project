import os
import random

import torch
from torch.utils.data import DataLoader

from config import (
    batch_size,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    learning_rate,
    device,
    data_path,
)

from dataset import (
    CharTokenizer,
    NextCharacterDataset,
    load_text_data,
    split_train_val,
)

from model import TinyGPT


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters: int = 20):
    model.eval()
    result = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []

        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break

            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses.append(loss.item())

        result[split] = sum(losses) / max(len(losses), 1)

    model.train()
    return result


def generate_sample(model, tokenizer, max_new_tokens: int = 300) -> str:
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens)

    return tokenizer.decode(generated[0].tolist())


def main():
    set_seed(42)

    checkpoints = [100, 500, 1000]
    total_train_steps = max(checkpoints)

    print("GPT 2.0 training experiment start")
    print(f"device: {device}")
    print(f"dataset path: {data_path}")

    text = load_text_data(data_path)

    print("\n[1] Dataset loaded")
    print(f"number of characters: {len(text):,}")

    tokenizer = CharTokenizer(text)
    encoded_data = tokenizer.encode(text)

    print("\n[2] Character tokenizer created")
    print(f"vocab size: {tokenizer.vocab_size}")

    train_data, val_data = split_train_val(encoded_data, train_ratio=0.9)

    train_dataset = NextCharacterDataset(train_data, block_size)
    val_dataset = NextCharacterDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    print("\n[3] Dataset split completed")
    print(f"train tokens: {len(train_data):,}")
    print(f"validation tokens: {len(val_data):,}")
    print(f"block size: {block_size}")
    print(f"batch size: {batch_size}")

    model = TinyGPT(tokenizer.vocab_size).to(device)

    num_params = sum(p.numel() for p in model.parameters())

    print("\n[4] TinyGPT model created")
    print(f"number of parameters: {num_params:,}")
    print(f"n_embd: {n_embd}")
    print(f"n_head: {n_head}")
    print(f"n_layer: {n_layer}")
    print(f"dropout: {dropout}")
    print(f"learning rate: {learning_rate}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    initial_losses = estimate_loss(model, train_loader, val_loader)

    print("\n[5] Initial loss")
    print(
        f"initial train loss {initial_losses['train']:.4f}, "
        f"initial val loss {initial_losses['val']:.4f}"
    )

    print("\n[6] Training experiment")
    print("checkpoints: 100, 500, 1000 steps")

    train_iter = iter(train_loader)
    interval_loss_sum = 0.0
    interval_step_count = 0

    os.makedirs("outputs", exist_ok=True)

    for step in range(1, total_train_steps + 1):
        try:
            x_batch, y_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_batch, y_batch = next(train_iter)

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits, loss = model(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        interval_loss_sum += loss.item()
        interval_step_count += 1

        if step in checkpoints:
            losses = estimate_loss(model, train_loader, val_loader)

            print()
            print(
                f"step {step:04d} | "
                f"interval train loss {interval_loss_sum / interval_step_count:.4f} | "
                f"eval train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            torch.manual_seed(42)
            generated_text = generate_sample(model, tokenizer, max_new_tokens=300)

            print(f"\nGenerated text after {step} training steps:")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)

            output_path = f"outputs/generated_step_{step}.txt"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(generated_text)

            print(f"generated text saved to: {output_path}")

            interval_loss_sum = 0.0
            interval_step_count = 0

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = "checkpoints/tiny_gpt_experiment.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "chars": tokenizer.chars,
            "checkpoints": checkpoints,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
        },
        checkpoint_path,
    )

    print("\n[7] Experiment completed")
    print(f"model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()