import os

import torch
from torch.utils.data import DataLoader

from config import (
    batch_size,
    block_size,
    max_iters,
    eval_interval,
    eval_iters,
    learning_rate,
    device,
    data_path,
    checkpoint_path,
)
from dataset import (
    CharTokenizer,
    NextCharacterDataset,
    load_text_data,
    split_train_val,
)
from model import TinyGPT


def estimate_loss(model, train_loader, val_loader):
    """
    train loss와 validation loss를 계산한다.

    학습 중 일정 step마다 현재 모델이 train data와 validation data에서
    어느 정도 loss를 보이는지 확인하기 위한 함수이다.
    """

    out = {}
    model.eval()

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []

        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break

            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    print("GPT 2.0 training start")
    print(f"device: {device}")

    # ---------------------------------------------------------
    # 1. Load custom dataset
    # ---------------------------------------------------------
    text = load_text_data(data_path)

    print("\n[1] Dataset loaded")
    print(f"dataset path: {data_path}")
    print(f"number of characters: {len(text):,}")

    # ---------------------------------------------------------
    # 2. Character-level tokenization
    # ---------------------------------------------------------
    tokenizer = CharTokenizer(text)
    encoded_data = tokenizer.encode(text)

    print("\n[2] Character tokenizer created")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"vocab: {''.join(tokenizer.chars[:100])}")

    # ---------------------------------------------------------
    # 3. Train / validation split
    # ---------------------------------------------------------
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
        shuffle=True,
        drop_last=True,
    )

    print("\n[3] Dataset split completed")
    print(f"train tokens: {len(train_data):,}")
    print(f"validation tokens: {len(val_data):,}")
    print(f"block size: {block_size}")
    print(f"batch size: {batch_size}")

    # ---------------------------------------------------------
    # 4. Build model
    # ---------------------------------------------------------
    model = TinyGPT(tokenizer.vocab_size).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print("\n[4] TinyGPT model created")
    print(f"number of parameters: {num_params:,}")

    # ---------------------------------------------------------
    # 5. Train model
    # ---------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("\n[5] Training")
    train_iter = iter(train_loader)

    for step in range(max_iters + 1):
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

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

    # ---------------------------------------------------------
    # 6. Save checkpoint
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "chars": tokenizer.chars,
    }

    torch.save(checkpoint, checkpoint_path)

    print("\n[6] Training completed")
    print(f"model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()