# ---------------------------------------------------------
# Model configuration
# notebook_06의 Tiny GPT 구조를 기준으로 하되,
# Codespaces에서 실행 가능하도록 크기를 적당히 줄였다.
# ---------------------------------------------------------

batch_size = 32
block_size = 64

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2


# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------

max_iters = 1000
eval_interval = 100
eval_iters = 100
learning_rate = 3e-4


# ---------------------------------------------------------
# Text generation configuration
# ---------------------------------------------------------

max_new_tokens = 500
temperature = 0.8
top_k = None


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

data_path = "data/input.txt"
checkpoint_path = "checkpoints/tiny_gpt.pt"


# ---------------------------------------------------------
# Device configuration
# ---------------------------------------------------------

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"