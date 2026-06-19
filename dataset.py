import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """
    Character-level tokenizer.

    수업 notebook의 흐름처럼 텍스트를 문자 단위로 나누고,
    각 문자를 정수 index로 변환한다.

    예:
        "abc" -> [0, 1, 2]
    """

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.chars = chars
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])


class NextCharacterDataset(Dataset):
    """
    GPT-style next character prediction dataset.

    하나의 긴 text를 block_size 길이의 x와 y로 나눈다.

    x = 현재까지의 문자 sequence
    y = x보다 한 글자 뒤의 정답 sequence

    예:
        text = "hello"
        block_size = 3

        x = "hel"
        y = "ell"

    즉, 모델은 각 위치에서 다음 문자를 예측하도록 학습된다.
    """

    def __init__(self, encoded_data, block_size: int):
        self.data = torch.tensor(encoded_data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_text_data(data_path: str):
    """
    data/input.txt 파일을 읽어온다.
    """

    with open(data_path, "r", encoding="utf-8") as file:
        text = file.read()

    if len(text.strip()) == 0:
        raise ValueError("data/input.txt is empty. Please add a custom dataset first.")

    return text


def split_train_val(encoded_data, train_ratio: float = 0.9):
    """
    전체 데이터를 train set과 validation set으로 나눈다.
    """

    n = int(len(encoded_data) * train_ratio)

    train_data = encoded_data[:n]
    val_data = encoded_data[n:]

    return train_data, val_data