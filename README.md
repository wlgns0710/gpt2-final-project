# GPT 2.0 Final Project

## 1. 프로젝트 개요

이 프로젝트는 PyTorch를 이용하여 작은 GPT-style character-level language model을 직접 구현한 프로젝트이다.

프로젝트의 목표는 `tiny Shakespeare`가 아닌 다른 데이터셋을 사용하여 GPT 2.0 형태의 모델을 학습시키고, 학습 결과와 text generation 결과를 제시하는 것이다.

본 프로젝트는 수업에서 다룬 notebook 흐름을 바탕으로 구성하였다.

```text
Bigram Language Model
→ MLP Character Model
→ GPT-style Sequence Dataset
→ Masked Self-Attention
→ Multi-Head Attention
→ FeedForward Network
→ Transformer Block
→ Tiny GPT Language Model
```

기존 GPT-from-scratch 예제에서 자주 사용하는 `tiny Shakespeare` 데이터셋은 사용하지 않았고, 직접 구성한 영어 영화 리뷰 스타일의 custom dataset을 사용하였다.

---

## 2. 프로젝트 목표

이 프로젝트의 주요 목표는 다음과 같다.

1. PyTorch를 이용하여 GPT-style language model을 직접 구현한다.
2. `tiny Shakespeare`가 아닌 다른 데이터셋으로 모델을 학습한다.
3. character-level tokenizer를 이용하여 문자를 정수 index로 변환한다.
4. 모델이 이전 문맥을 보고 다음 character를 예측하도록 학습한다.
5. 학습 과정에서 train loss와 validation loss를 확인한다.
6. 학습된 모델을 이용하여 새로운 text를 생성한다.
7. 코드 구조를 파일별로 분리하여 전체 작동 과정을 이해하기 쉽게 정리한다.

---

## 3. 데이터셋

### 3.1 데이터셋 설명

데이터셋은 다음 파일에 저장하였다.

```text
data/input.txt
```

본 프로젝트에서는 영어 영화 리뷰 스타일의 custom text dataset을 사용하였다.
데이터셋은 영화의 분위기, 인물 변화, 갈등, 음악, 촬영, 결말, 감정 해석 등을 설명하는 문단들로 구성되어 있다.

이 데이터셋은 `tiny Shakespeare`가 아니며, GPT 2.0 프로젝트 요구사항에 맞게 별도로 구성한 텍스트 데이터셋이다.

### 3.2 데이터셋 정보

학습 실행 시 출력된 데이터셋 정보는 다음과 같다.

```text
number of characters: 6,666
vocab size: 46
```

vocabulary 출력 예시는 다음과 같다.

```text
 ,-.:ABEFHIMNOPRSTUabcdefghijklmnopqrstuvwxyz
```

이 프로젝트는 character-level tokenizer를 사용한다.
즉, 단어 단위가 아니라 문자 단위로 텍스트를 나누고, 각 문자를 정수 index로 변환한다.

예를 들어 다음 문장이 있다고 하면,

```text
The movie
```

모델 내부에서는 각 문자가 정수 index로 변환되어 처리된다.

---

## 4. 프로젝트 구조

```text
gpt2-final-project/
├── README.md
├── requirements.txt
├── config.py
├── dataset.py
├── model.py
├── train.py
├── generate.py
├── data/
│   └── input.txt
└── checkpoints/
```

| 파일                 | 역할                                        |
| ------------------ | ----------------------------------------- |
| `README.md`        | 프로젝트 설명 및 실행 결과 정리                        |
| `requirements.txt` | 실행에 필요한 Python package 목록                 |
| `config.py`        | hyperparameter와 파일 경로 설정                  |
| `dataset.py`       | character tokenizer와 sequence dataset 구현  |
| `model.py`         | GPT-style model architecture 구현           |
| `train.py`         | 모델 학습 코드                                  |
| `generate.py`      | 학습된 모델을 이용한 text generation 코드            |
| `data/input.txt`   | custom English movie-review-style dataset |
| `checkpoints/`     | 학습된 model checkpoint 저장 폴더                |

---

## 5. 모델 구조

이 프로젝트의 모델은 character-level Tiny GPT이다.

모델은 문자 index sequence를 입력으로 받고, 각 위치에서 다음 character를 예측한다.

전체 구조는 다음과 같다.

```text
Token Embedding
→ Positional Embedding
→ Transformer Blocks
→ Final LayerNorm
→ Linear Language Modeling Head
```

각 Transformer block은 다음 구조로 이루어진다.

```text
LayerNorm
→ Multi-Head Masked Self-Attention
→ Residual Connection
→ LayerNorm
→ FeedForward Network
→ Residual Connection
```

즉, 수업에서 다룬 self-attention, multi-head attention, feedforward network, residual connection, layer normalization을 하나의 GPT-style model로 연결하였다.

---

## 6. 주요 구현 내용

### 6.1 Character Tokenizer

`dataset.py`에서는 character-level tokenizer를 구현하였다.

tokenizer는 두 개의 dictionary를 만든다.

```text
stoi: character → integer
itos: integer → character
```

코드에서는 다음과 같이 구현하였다.

```python
self.stoi = {ch: i for i, ch in enumerate(chars)}
self.itos = {i: ch for i, ch in enumerate(chars)}
```

이를 통해 text를 숫자 sequence로 바꾸고, 다시 숫자 sequence를 text로 복원할 수 있다.

### 6.2 GPT-style Dataset

모델은 다음 character prediction 방식으로 학습한다.

예를 들어 text가 다음과 같고,

```text
hello
```

`block_size = 3`이라면 하나의 학습 sample은 다음과 같이 구성될 수 있다.

```text
x = hel
y = ell
```

즉, `x`는 입력 sequence이고, `y`는 한 글자 뒤로 밀린 정답 sequence이다.

코드에서는 다음과 같이 구현하였다.

```python
x = self.data[idx : idx + self.block_size]
y = self.data[idx + 1 : idx + self.block_size + 1]
```

이 구조를 통해 모델은 각 위치에서 다음 문자를 예측하도록 학습된다.

### 6.3 Masked Self-Attention

모델은 masked self-attention을 사용한다.

self-attention은 현재 위치의 token이 이전 token들을 참고할 수 있게 해준다.
하지만 language model에서는 미래 token을 미리 보면 안 되기 때문에 causal mask를 적용한다.

causal mask는 lower triangular matrix로 만든다.

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

그리고 미래 위치를 다음과 같이 막는다.

```python
weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
```

이를 통해 현재 character는 자신보다 뒤에 있는 미래 character를 볼 수 없다.

### 6.4 Multi-Head Attention

하나의 attention head만 사용하는 대신, 여러 개의 attention head를 병렬로 사용하였다.

```python
self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
```

각 head는 서로 다른 관점에서 attention을 계산하고, 그 결과를 concat한 뒤 다시 projection한다.

### 6.5 FeedForward Network

attention 결과는 feedforward network를 통과한다.

```python
nn.Linear(n_embd, 4 * n_embd)
nn.ReLU()
nn.Linear(4 * n_embd, n_embd)
```

이 부분은 각 token position마다 독립적으로 적용되며, 모델에 비선형 변환 능력을 추가한다.

### 6.6 Residual Connection과 Layer Normalization

Transformer block에서는 residual connection과 layer normalization을 사용한다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

이 구조는 학습을 안정적으로 만들고, 여러 block을 쌓을 수 있게 해준다.

---

## 7. Hyperparameters

주요 hyperparameter는 `config.py`에 정리하였다.

```python
batch_size = 32
block_size = 128
max_iters = 1000
eval_interval = 100
eval_iters = 100
learning_rate = 3e-4

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
```

| Hyperparameter  | 의미                     |
| --------------- | ---------------------- |
| `batch_size`    | 한 번에 학습하는 sample 개수    |
| `block_size`    | 모델이 한 번에 보는 context 길이 |
| `max_iters`     | 전체 학습 반복 횟수            |
| `learning_rate` | optimizer의 학습률         |
| `n_embd`        | embedding dimension    |
| `n_head`        | attention head 개수      |
| `n_layer`       | Transformer block 개수   |
| `dropout`       | dropout 비율             |

모델 크기는 GitHub Codespaces 환경에서도 실행 가능하도록 비교적 작게 설정하였다.

---

## 8. 설치 방법

필요한 package는 다음 명령어로 설치한다.

```bash
pip install -r requirements.txt
```

`requirements.txt`에는 다음 package가 포함되어 있다.

```text
torch
numpy
```

`numpy`가 없어도 학습은 실행될 수 있지만, PyTorch 실행 중 경고가 발생할 수 있어 함께 포함하였다.

---

## 9. 학습 방법

모델 학습은 다음 명령어로 실행한다.

```bash
python train.py
```

`train.py`는 다음 순서로 작동한다.

```text
1. data/input.txt 불러오기
2. character-level tokenizer 생성
3. text를 integer id로 encoding
4. train / validation split
5. PyTorch Dataset과 DataLoader 생성
6. TinyGPT model 생성
7. AdamW optimizer로 학습
8. 일정 step마다 train loss와 validation loss 출력
9. checkpoints/model.pt에 checkpoint 저장
```

---

## 10. 학습 결과

custom movie-review-style dataset으로 모델을 학습하였다.

실행 결과는 다음과 같다.

```text
GPT 2.0 training start
device: cpu

[1] Dataset loaded
dataset path: data/input.txt
number of characters: 6,666

[2] Character tokenizer created
vocab size: 46
vocab: 
 ,-.:ABEFHIMNOPRSTUabcdefghijklmnopqrstuvwxyz

[3] Dataset split completed
train tokens: 5,999
validation tokens: 667
block size: 128
batch size: 32

[4] TinyGPT model created
number of parameters: 820,014

[5] Training
step 0: train loss 3.8384, val loss 3.8506
step 100: train loss 2.4002, val loss 2.4925
step 200: train loss 2.2280, val loss 2.3979
step 300: train loss 2.0424, val loss 2.3497
step 400: train loss 1.6723, val loss 2.3445
step 500: train loss 1.0253, val loss 2.3128
step 600: train loss 0.5215, val loss 2.5603
step 700: train loss 0.2829, val loss 2.8419
step 800: train loss 0.1923, val loss 3.0983
step 900: train loss 0.1521, val loss 3.2921
step 1000: train loss 0.1331, val loss 3.2886

[6] Training completed
model checkpoint saved to: checkpoints/model.pt
```

### 10.1 학습 결과 해석

train loss는 `3.8384`에서 `0.1331`까지 감소하였다.

이는 모델이 custom dataset의 문자 패턴과 문장 구조를 학습했다는 것을 보여준다.

반면 validation loss는 일정 시점 이후 증가하였다.
이는 데이터셋 크기가 작고, 모델이 train data를 강하게 학습하면서 overfitting이 발생했기 때문으로 볼 수 있다.

즉, 현재 모델은 학습 데이터의 스타일과 문장 패턴은 잘 따라가지만, 새로운 validation data에 대한 일반화 성능은 제한적이다.

이 결과는 작은 custom dataset으로 character-level GPT를 학습했을 때 나타날 수 있는 자연스러운 결과이다.

---

## 11. Text Generation

학습된 모델로 text를 생성하려면 다음 명령어를 실행한다.

```bash
python generate.py
```

`generate.py`는 다음 checkpoint를 불러온다.

```text
checkpoints/model.pt
```

이후 모델은 이전까지 생성된 character sequence를 바탕으로 다음 character를 하나씩 sampling하여 새로운 text를 생성한다.

---

## 12. 생성 결과

생성 결과는 다음과 같다.

```text
The main character is not perfect. The film shows that reconciliation is difficult because memory is never the same for everyone.

The ending is parts of the move is its atmosphere. The music does not overwhelm the scenes. It stays behind the characters and supports their emotions. The music is s not only about crime. It is also about fear, guilt, and the choices people make when they want to protect themselves.

pach rterematin than theroxagge parts of thile becomes. He e is afraid, makes mista
```

### 12.1 생성 결과 해석

생성 결과의 앞부분은 비교적 자연스럽다.

예를 들어 다음 문장은 custom movie review dataset의 스타일을 잘 따라간다.

```text
The main character is not perfect. The film shows that reconciliation is difficult because memory is never the same for everyone.
```

또한 다음과 같은 영화 리뷰식 표현도 생성되었다.

```text
The music does not overwhelm the scenes.
```

그러나 뒤로 갈수록 문장이 불안정해지고, 의미가 불명확한 문자열이 나타난다.

```text
pach rterematin than theroxagge parts of thile becomes
```

이는 모델이 character-level 방식으로 학습되었고, 데이터셋 크기가 작기 때문에 발생한 한계이다.

그래도 생성 결과를 보면 모델이 일부 단어, 표현, 문장 분위기, 영화 리뷰 스타일을 학습했다는 것을 확인할 수 있다.

---

## 13. tiny Shakespeare와의 차이점

기존 GPT-from-scratch 예제에서는 `tiny Shakespeare` 데이터셋을 자주 사용한다.

하지만 이 프로젝트에서는 `tiny Shakespeare`를 사용하지 않았다.

대신 직접 구성한 영어 영화 리뷰 스타일의 custom dataset을 사용하였다.

| 항목           | 본 프로젝트                            |
| ------------ | --------------------------------- |
| 데이터셋         | Custom movie-review-style text    |
| 언어           | English                           |
| Tokenization | Character-level                   |
| 모델           | Tiny GPT                          |
| 학습 목표        | Next character prediction         |
| 생성 결과        | Movie-review-style generated text |

따라서 본 프로젝트는 “tiny Shakespeare 이외의 다른 데이터셋으로 GPT 2.0을 학습하고 결과를 제시하라”는 조건을 충족한다.

---

## 14. 한계점 및 개선 방향

현재 모델에는 다음과 같은 한계가 있다.

1. 데이터셋 크기가 작다.
2. character-level tokenizer를 사용하기 때문에 긴 문장 생성이 불안정하다.
3. 생성 길이가 길어질수록 문법과 의미가 무너지는 경우가 있다.
4. validation loss가 후반부에 증가하여 overfitting이 확인된다.
5. 이 모델은 대형 언어모델처럼 의미를 깊게 이해하는 모델이 아니라, 학습 데이터의 문자 패턴을 바탕으로 다음 문자를 예측하는 작은 GPT-style model이다.

개선 방향은 다음과 같다.

1. 데이터셋 크기를 늘린다.
2. 더 다양한 장르의 텍스트를 추가한다.
3. early stopping을 적용하여 overfitting을 줄인다.
4. character-level tokenizer 대신 subword tokenizer를 사용한다.
5. model size와 dropout 값을 조정한다.
6. 더 긴 시간 동안 GPU 환경에서 학습한다.

---

## 15. 실행 재현 방법

### 15.1 Package 설치

```bash
pip install -r requirements.txt
```

### 15.2 모델 학습

```bash
python train.py
```

### 15.3 Text generation

```bash
python generate.py
```

`checkpoints/` 폴더는 `.gitignore`에 포함되어 있으므로 GitHub에는 checkpoint 파일을 업로드하지 않는다.
따라서 text generation을 실행하려면 먼저 `python train.py`로 모델을 학습해야 한다.

---

## 16. 참고 자료

이 프로젝트는 수업 notebook과 GPT-from-scratch 구조를 참고하여 구현하였다.

주요 참고 자료는 다음과 같다.

* Course notebooks: `notebook_01` to `notebook_06`
* `torch_dataset` notebook
* Andrej Karpathy, Neural Networks: Zero to Hero
* GPT-style Transformer language modeling structure
