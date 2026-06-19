import torch


# ---------------------------------------------------------
# Device setting
# ---------------------------------------------------------
# GPU가 있으면 cuda를 사용하고, 없으면 cpu를 사용한다.
# Codespaces 환경에서는 보통 cpu로 실행된다.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------
# tiny Shakespeare를 사용하지 않고, 직접 구성한 custom dataset을 사용한다.
data_path = "data/input.txt"

# 학습이 끝난 model parameter를 저장할 위치
checkpoint_path = "checkpoints/model.pt"


# ---------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------
# 수업 notebook_04, notebook_05, notebook_06의 흐름처럼
# text를 block 단위로 잘라 next character prediction을 수행한다.

batch_size = 32
block_size = 128

max_iters = 1000
eval_interval = 100
eval_iters = 100

learning_rate = 3e-4


# ---------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------
# notebook_06의 Tiny GPT 구조를 기준으로 하되,
# Codespaces에서 실행 가능하도록 크기를 적당히 줄였다.

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2


# ---------------------------------------------------------
# Text generation setting
# ---------------------------------------------------------
max_new_tokens = 300