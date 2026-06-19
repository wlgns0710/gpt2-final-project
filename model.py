import torch
import torch.nn as nn
import torch.nn.functional as F

from config import block_size, n_embd, n_head, n_layer, dropout, device


class Head(nn.Module):
    """
    Single masked self-attention head.

    notebook_05에서 다룬 self-attention 구조를 코드로 구현한 부분이다.
    각 token은 query, key, value vector로 변환되고,
    현재 token은 미래 token을 볼 수 없도록 lower triangular mask를 적용한다.
    """

    def __init__(self, head_size: int):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, time_steps, channels = x.shape

        k = self.key(x)
        q = self.query(x)

        # attention score 계산
        # q @ k.T / sqrt(head_size)
        weights = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # 미래 token을 보지 못하도록 masking
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))

        # attention distribution
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        output = weights @ v

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi-head masked self-attention.

    여러 개의 attention head를 병렬로 실행한 뒤,
    결과를 concat하고 linear projection을 적용한다.
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    attention 결과를 각 token position마다 독립적으로 변환한다.
    notebook_06의 FeedForward 구조를 따른다.
    """

    def __init__(self, n_embd: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block.

    하나의 block은 다음 구조로 이루어진다.

    LayerNorm
    → MultiHeadAttention
    → Residual Connection
    → LayerNorm
    → FeedForward
    → Residual Connection
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class TinyGPT(nn.Module):
    """
    Character-level Tiny GPT model.

    notebook_06의 Tiny GPT 구조를 기반으로 구현하였다.

    입력:
        idx: character index sequence

    출력:
        logits: 각 위치에서 다음 character에 대한 score
        loss: 정답 targets가 주어졌을 때 cross entropy loss
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Transformer에서 자주 사용하는 방식으로 weight를 초기화한다.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch_size, time_steps = idx.shape

        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(
            torch.arange(time_steps, device=device)
        )

        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, time_steps, channels = logits.shape

            logits = logits.view(batch_size * time_steps, channels)
            targets = targets.view(batch_size * time_steps)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens: int):
        """
        학습된 모델을 이용하여 새로운 text를 생성한다.

        현재까지 생성된 idx sequence를 보고,
        다음 character를 하나씩 sampling하여 이어 붙인다.
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx