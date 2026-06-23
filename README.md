# GPT 2.0 Final Project

## 1. Project Overview

이 프로젝트는 인공지능과 금융공학 수업에서 학습한 내용을 바탕으로, 작은 규모의 GPT-style language model을 직접 구현한 프로젝트이다.

실제 OpenAI GPT-2 전체 모델을 그대로 재현한 것은 아니며, 수업에서 다룬 GPT 구조의 핵심 요소를 직접 코드로 구현하고, custom text dataset을 이용해 character-level language model을 학습시키는 것을 목표로 하였다.

이 프로젝트에서 구현한 핵심 요소는 다음과 같다.

* next-character prediction
* character-level tokenization
* token embedding
* positional embedding
* masked self-attention
* single self-attention head
* multi-head attention
* feedforward network
* transformer block
* residual connection
* LayerNorm
* multiple stacked transformer blocks
* language modeling head
* autoregressive text generation
* train / validation loss 출력
* checkpoint 저장 및 불러오기

최종적으로 모델은 tiny Shakespeare가 아닌 custom movie-review-style dataset을 학습하고, 학습 후 새로운 movie-review-style text를 생성하였다.

---

## 2. Project Structure

```text
gpt2-final-project/
├── README.md
├── requirements.txt
├── config.py
├── dataset.py
├── model.py
├── train.py
├── generate.py
├── main.py
├── check_project.py
├── data/
│   └── input.txt
├── outputs/
│   ├── generated_step_100.txt
│   ├── generated_step_500.txt
│   └── generated_step_1000.txt
└── checkpoints/
    └── tiny_gpt.pt
```

---

## 3. File Description

| File               | Description                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`        | 모델 크기, 학습 설정, 파일 경로, device 설정을 관리한다.                                                                                                                |
| `dataset.py`       | text loading, character-level tokenizer, train / validation split, next-character dataset 구성을 담당한다.                                                  |
| `model.py`         | GPT-style decoder-only Transformer 모델을 구현한다. masked self-attention, multi-head attention, feedforward network, residual connection, LayerNorm을 포함한다. |
| `train.py`         | 기본 학습 파일이다. custom dataset을 불러와 TinyGPT 모델을 학습하고 checkpoint를 저장한다.                                                                                   |
| `generate.py`      | 저장된 checkpoint를 불러와 autoregressive 방식으로 text를 생성한다.                                                                                                  |
| `main.py`          | 100 / 500 / 1000 training steps에서 loss와 generated text를 비교하는 실험 파일이다.                                                                                |
| `check_project.py` | dataset, tokenizer, model forward pass, generate 함수가 정상 작동하는지 확인하는 검사 파일이다.                                                                          |
| `data/input.txt`   | tiny Shakespeare가 아닌 직접 구성한 custom movie-review-style dataset이다.                                                                                     |
| `outputs/`         | training step별 generated text 결과를 저장한다.                                                                                                              |
| `checkpoints/`     | 학습된 모델 checkpoint를 저장한다.                                                                                                                             |

---

## 4. Custom Dataset

공지에 따라 최종 학습에는 tiny Shakespeare가 아닌 custom dataset을 사용하였다.

```text
data/input.txt
```

이번 프로젝트에서는 영화 리뷰 스타일의 영어 문장들을 직접 구성하여 dataset으로 사용하였다.

dataset은 영화의 분위기, 인물, 기억, 갈등, 결말, 감정, 시각적 연출 등을 설명하는 문장들로 구성하였다.

이 dataset을 선택한 이유는 다음과 같다.

1. tiny Shakespeare가 아닌 별도의 dataset을 사용하기 위함이다.
2. 문장 구조가 비교적 단순하여 작은 GPT 모델이 학습하기 적합하다.
3. `The film`, `The story`, `The character`, `The ending`, `memory`, `audience` 등 반복되는 표현이 있어 character-level model이 문체를 학습하는지 확인하기 좋다.
4. 생성 결과에서 영화 리뷰와 비슷한 표현이 나타나는지 비교하기 쉽다.

최종 실행 결과 dataset의 크기는 다음과 같다.

```text
number of characters: 11,479
vocab size: 48
```

---

## 5. Character-level Tokenization

이 프로젝트는 단어 단위가 아니라 문자 하나를 하나의 token으로 사용하는 character-level language model이다.

예를 들어 다음과 같은 문장이 있다고 하면,

```text
The film
```

모델은 이를 단어 단위로 나누지 않고, 문자 단위로 나누어 처리한다.

```text
T, h, e,  , f, i, l, m
```

`CharTokenizer`는 dataset에 등장하는 고유 문자들을 모아 vocabulary를 만들고, 각 문자에 정수 index를 부여한다.

```python
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

여기서 각 mapping의 의미는 다음과 같다.

| Mapping | Meaning                                   |
| ------- | ----------------------------------------- |
| `stoi`  | string to integer. 문자를 정수 token으로 변환한다.   |
| `itos`  | integer to string. 정수 token을 다시 문자로 변환한다. |

Encoding은 문자열을 정수 token sequence로 변환하는 과정이고, decoding은 정수 token sequence를 다시 문자열로 변환하는 과정이다.

최종 학습에서 사용된 vocabulary는 다음과 같다.

```text
 ,-.:ABEFHILMNOPRSTUWabcdefghijklmnopqrstuvwxyz
```

---

## 6. Next-character Prediction Dataset

GPT-style language model은 이전 token들을 바탕으로 다음 token을 예측한다.

이 프로젝트에서는 character-level model이므로, 정확히는 next-character prediction을 수행한다.

`NextCharacterDataset`에서는 input sequence `x`와 target sequence `y`를 한 글자씩 밀어서 구성한다.

```python
x = data[idx : idx + block_size]
y = data[idx + 1 : idx + block_size + 1]
```

예를 들어 text가 다음과 같다고 하자.

```text
hello
```

`block_size = 4`인 경우 input과 target은 다음과 같이 구성된다.

```text
x: h e l l
y: e l l o
```

즉, 모델은 각 위치에서 다음 문자를 예측하도록 학습된다.

이 프로젝트에서는 다음 설정을 사용하였다.

```python
block_size = 64
```

따라서 모델은 한 번에 최대 64개의 이전 문자를 context로 사용하여 다음 문자를 예측한다.

최종 dataset split 결과는 다음과 같다.

```text
train tokens: 10,331
validation tokens: 1,148
block size: 64
batch size: 32
```

---

## 7. Model Architecture

이 프로젝트의 모델은 decoder-only Transformer 구조를 따른다.

전체 구조는 다음과 같다.

```text
Token indices
      ↓
Token embedding
      +
Positional embedding
      ↓
Transformer blocks
      ↓
Final LayerNorm
      ↓
Linear output layer
      ↓
Vocabulary logits
```

모델은 입력 token index를 embedding vector로 변환한 뒤, positional embedding을 더한다.

```python
tok_emb = self.token_embedding_table(idx)
pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
x = tok_emb + pos_emb
```

Token embedding은 각 문자 token을 학습 가능한 vector로 변환한다.

Positional embedding은 각 token이 sequence 안에서 몇 번째 위치에 있는지 알려준다.

Self-attention은 token 간의 관계를 계산할 수 있지만, 기본적으로 순서 정보를 직접 알지는 못한다. 따라서 positional embedding을 더해 sequence의 위치 정보를 모델에 제공한다.

---

## 8. Masked Self-Attention

GPT는 autoregressive model이다.

즉, 현재 위치에서 다음 token을 예측할 때 미래 token을 보면 안 된다.

예를 들어 다음 문장을 예측한다고 하자.

```text
The film
```

모델이 `f` 위치 이후의 문자를 예측할 때, 그 뒤에 나올 미래 문자들을 미리 보면 안 된다.

이를 막기 위해 masked self-attention을 사용한다.

이 프로젝트에서는 lower triangular matrix를 이용하여 미래 위치를 가린다.

```python
self.register_buffer(
    "tril",
    torch.tril(torch.ones(block_size, block_size))
)
```

Attention score를 계산한 뒤, 미래 위치에 해당하는 부분을 `-inf`로 바꾼다.

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

이후 softmax를 적용하면 미래 token에 대한 attention weight는 0이 된다.

Scaled dot-product attention은 다음과 같다.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

각 vector의 의미는 다음과 같다.

| Vector | Meaning                  |
| ------ | ------------------------ |
| Query  | 현재 token이 찾고자 하는 정보      |
| Key    | 각 token이 가진 특징           |
| Value  | attention을 통해 실제로 가져올 정보 |

`QK^T`를 `sqrt(d_k)`로 나누는 이유는 dot product 값이 지나치게 커지는 것을 막기 위해서이다.

값이 너무 커지면 softmax가 한쪽으로 과하게 쏠릴 수 있기 때문에 scaling을 적용한다.

---

## 9. Single Self-Attention Head

Single self-attention head는 하나의 관점에서 sequence 안의 token 관계를 계산한다.

각 token representation에서 query, key, value를 만든다.

```python
q = self.query(x)
k = self.key(x)
v = self.value(x)
```

그 다음 query와 key의 내적을 이용해 attention score를 계산한다.

```python
wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
```

causal mask를 적용한 뒤 softmax를 사용하여 attention weight를 만든다.

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)
```

마지막으로 attention weight와 value를 곱하여 head의 output을 만든다.

```python
out = wei @ v
```

이 과정이 하나의 self-attention head에서 수행된다.

---

## 10. Multi-Head Attention

Multi-head attention은 여러 개의 attention head를 병렬로 사용하는 구조이다.

하나의 attention head만 사용하면 token 관계를 한 가지 관점에서만 볼 수 있다.

반면 여러 head를 사용하면 서로 다른 head가 서로 다른 관계를 학습할 수 있다.

예를 들어 어떤 head는 문장 초반과 후반의 관계를 볼 수 있고, 다른 head는 반복되는 표현이나 punctuation 주변의 패턴을 학습할 수 있다.

이 프로젝트에서는 다음 설정을 사용하였다.

```python
n_head = 4
```

즉, 총 4개의 attention head를 사용한다.

각 head의 출력은 concatenate된 뒤 linear projection을 통과한다.

```python
out = torch.cat([h(x) for h in self.heads], dim=-1)
out = self.proj(out)
```

구술 테스트에서 이 부분은 다음 흐름으로 설명할 수 있다.

```text
single attention head를 여러 개 병렬로 사용하고,
각 head의 출력을 concat한 뒤 projection layer를 통과시키면 multi-head attention이 된다.
```

---

## 11. Feedforward Network

Attention layer는 token 사이의 관계를 계산한다.

그 다음에는 feedforward network를 사용하여 각 token representation을 비선형적으로 변환한다.

이 프로젝트의 feedforward network는 다음과 같은 구조를 가진다.

```python
nn.Linear(n_embd, 4 * n_embd)
nn.ReLU()
nn.Linear(4 * n_embd, n_embd)
nn.Dropout(dropout)
```

중간 차원을 `4 * n_embd`로 확장한 뒤 다시 `n_embd`로 줄인다.

이는 Transformer에서 자주 사용되는 구조이다.

---

## 12. Transformer Block

Transformer block은 multi-head attention과 feedforward network를 하나로 묶은 구조이다.

이 프로젝트에서는 하나의 block 안에 다음 요소들이 포함된다.

1. LayerNorm
2. Multi-head self-attention
3. Residual connection
4. LayerNorm
5. Feedforward network
6. Residual connection

구조는 다음과 같다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

첫 번째 줄에서는 LayerNorm을 통과한 입력이 multi-head attention을 거친 뒤, 원래 입력 `x`와 더해진다.

두 번째 줄에서는 다시 LayerNorm을 통과한 입력이 feedforward network를 거친 뒤, 원래 입력과 더해진다.

---

## 13. Residual Connection and LayerNorm

Transformer block에서는 residual connection과 LayerNorm을 사용한다.

Residual connection은 기존 입력 `x`를 보존하면서 attention 또는 feedforward의 결과를 더한다.

이를 통해 깊은 모델에서도 gradient가 더 잘 전달될 수 있다.

LayerNorm은 각 token representation의 분포를 정규화하여 학습을 안정적으로 만든다.

이 프로젝트에서는 sublayer를 통과하기 전에 LayerNorm을 먼저 적용하는 pre-norm 구조를 사용하였다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

---

## 14. Stacked Transformer Blocks and Language Modeling Head

GPT 구조는 transformer block 하나만 사용하는 것이 아니라, 여러 transformer block을 쌓아서 만든다.

이 프로젝트에서는 다음 설정을 사용하였다.

```python
n_layer = 4
```

즉, transformer block 4개를 순서대로 쌓았다.

```python
self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
```

여러 block을 통과한 후 final LayerNorm을 적용하고, 마지막 linear layer를 통해 vocabulary size만큼의 logits를 출력한다.

```python
x = self.blocks(x)
x = self.ln_f(x)
logits = self.lm_head(x)
```

이 `lm_head`가 language modeling head이다.

각 위치에서 다음 문자가 vocabulary의 어떤 문자일지 예측한다.

---

## 15. Output Shape and Loss

모델은 각 위치에서 다음 문자가 vocabulary의 어떤 문자일지 예측한다.

일반적인 logits 형태는 다음과 같다.

```text
B x T x C
```

각 문자의 의미는 다음과 같다.

| Symbol | Meaning         |
| ------ | --------------- |
| `B`    | batch size      |
| `T`    | sequence length |
| `C`    | vocabulary size |

Cross entropy loss 계산을 위해 batch dimension과 sequence dimension을 합친 형태로 logits를 변환한다.

```python
B, T, C = logits.shape
logits = logits.view(B * T, C)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

`F.cross_entropy`는 내부적으로 softmax와 negative log likelihood 계산을 포함한다.

따라서 학습 과정에서 logits에 softmax를 미리 적용하지 않는다.

---

## 16. Training Step

한 training step은 하나의 batch에 대해 다음 과정이 한 번 수행되는 것을 의미한다.

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

각 단계의 의미는 다음과 같다.

| Step               | Meaning                                    |
| ------------------ | ------------------------------------------ |
| `zero_grad()`      | 이전 step에서 계산된 gradient를 초기화한다.             |
| `loss.backward()`  | 현재 loss에 대한 gradient를 계산한다.                |
| `optimizer.step()` | 계산된 gradient를 이용해 model parameter를 업데이트한다. |

Optimizer는 AdamW를 사용하였다.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

---

## 17. Hyperparameters

이번 프로젝트에서 사용한 주요 hyperparameter는 다음과 같다.

| Hyperparameter  |  Value |
| --------------- | -----: |
| `batch_size`    |     32 |
| `block_size`    |     64 |
| `n_embd`        |    128 |
| `n_head`        |      4 |
| `n_layer`       |      4 |
| `dropout`       |    0.2 |
| `learning_rate` | 0.0003 |
| `max_iters`     |   1000 |
| `device`        |    cpu |
| `optimizer`     |  AdamW |

Codespaces 환경에서 실행 가능하도록 모델 크기를 너무 크게 잡지 않았다.

따라서 전체 parameter 수는 약 81만 개 정도로 유지하였다.

최종 학습 실행에서 확인한 parameter 수는 다음과 같다.

```text
number of parameters: 812,336
```

---

## 18. Project Check

본격적인 학습 전에 `check_project.py`를 작성하여 프로젝트 구성 요소가 정상적으로 작동하는지 확인하였다.

실행 명령어는 다음과 같다.

```bash
python check_project.py
```

이 검사를 통해 다음 요소들이 정상 작동함을 확인하였다.

| Check Item                    | Result  |
| ----------------------------- | ------- |
| config import                 | Success |
| dataset loading               | Success |
| tokenizer encoding / decoding | Success |
| train / validation split      | Success |
| dataset indexing              | Success |
| model creation                | Success |
| forward pass                  | Success |
| loss calculation              | Success |
| generate function             | Success |

Project check의 목적은 전체 학습 전에 dataset, tokenizer, model, loss, generation pipeline이 정상적으로 연결되어 있는지 확인하는 것이다.

생성 결과는 학습 전 모델에서는 무작위 문자에 가깝게 나온다. 이는 정상적인 결과이며, 이후 training을 통해 dataset의 패턴을 학습하게 된다.

---

## 19. Final Training Result

최종 학습은 expanded custom movie-review-style dataset을 사용하여 진행하였다.

실행 명령어는 다음과 같다.

```bash
python train.py
```

Codespaces 환경에서는 CUDA가 사용 가능하지 않았기 때문에 CPU로 학습을 진행하였다.

실행 결과는 다음과 같다.

```text
GPT 2.0 training start
device: cpu

[1] Dataset loaded
dataset path: data/input.txt
number of characters: 11,479

[2] Character tokenizer created
vocab size: 48
vocab: 
 ,-.:ABEFHILMNOPRSTUWabcdefghijklmnopqrstuvwxyz

[3] Dataset split completed
train tokens: 10,331
validation tokens: 1,148
block size: 64
batch size: 32

[4] TinyGPT model created
number of parameters: 812,336

[5] Training
step 0: train loss 3.9340, val loss 3.9445
step 100: train loss 2.4046, val loss 2.4055
step 200: train loss 2.2195, val loss 2.2543
step 300: train loss 2.0414, val loss 2.1166
step 400: train loss 1.6805, val loss 1.8125
step 500: train loss 1.3791, val loss 1.6119
step 600: train loss 1.1458, val loss 1.5180
step 700: train loss 0.9675, val loss 1.5103
step 800: train loss 0.8149, val loss 1.5244
step 900: train loss 0.6855, val loss 1.5558
step 1000: train loss 0.5865, val loss 1.6293

[6] Training completed
model checkpoint saved to: checkpoints/tiny_gpt.pt
```

Training loss는 `3.9340`에서 `0.5865`까지 감소하였다.

이는 모델이 custom dataset의 character-level pattern을 학습했음을 보여준다.

Validation loss는 초반과 중반에는 감소했지만, 후반에는 약간 증가하였다.

이는 dataset 크기가 작기 때문에 학습 후반부에서 overfitting이 발생하기 시작했을 가능성을 보여준다.

---

## 20. Training Result Interpretation

이번 학습 결과를 통해 두 가지를 확인할 수 있었다.

첫째, Tiny GPT 모델은 정상적으로 학습되었다.

Training loss가 꾸준히 감소했기 때문에, 모델이 학습 데이터의 문자 패턴을 점점 더 잘 예측하게 되었음을 알 수 있다.

둘째, 작은 custom dataset에서는 overfitting이 발생할 수 있다.

Validation loss는 step 700 부근까지 감소하다가 이후 약간 증가하였다.

이는 모델이 training dataset에는 계속 더 잘 맞춰졌지만, validation data에 대해서는 일정 시점 이후 일반화 성능이 조금 낮아졌음을 의미한다.

따라서 이번 실험에서는 모델이 정상적으로 학습되는 동시에, dataset size가 language model의 generalization에 중요한 영향을 준다는 점도 확인할 수 있었다.

---

## 21. Text Generation Result

학습이 끝난 후 저장된 checkpoint를 불러와 text generation을 실행하였다.

실행 명령어는 다음과 같다.

```bash
python generate.py
```

실행 결과는 다음과 같다.

```text
[1] Checkpoint loaded
checkpoint path: checkpoints/tiny_gpt.pt
vocab size: 48

[2] Model loaded

[3] Generated text
------------------------------------------------------------
The movie is not explain he audience tharacters through action always make memory. The characters detailence to tways feel him eanin the audienceobs or dramatic otion the explain uncomes.

The clont expries, makes the ending in of the ark, or a prossssible.

The film creates scene. They is makfod, and reciding more not overwards ama bigggertzes that the characters are lialso wling exsplain withoug explare he audience feel becomes magine iquiet lattern because, and se oof the story forgive every tryon a s
```

생성 결과는 완벽한 영어 문장은 아니지만, movie-review-style dataset과 관련된 단어와 표현을 포함한다.

예를 들어 다음과 같은 단어들이 생성 결과에 나타났다.

```text
movie
audience
characters
memory
ending
film
scene
story
```

이는 모델이 완전히 무작위 문자를 생성하는 것이 아니라, dataset에서 반복적으로 등장한 주제와 표현을 어느 정도 학습했음을 보여준다.

다만 일부 단어는 깨지거나 문법적으로 어색하다.

이는 모델이 작은 dataset을 기반으로 한 character-level language model이기 때문이다.

Character-level model은 단어를 직접 token으로 학습하지 않고, 문자 하나하나를 예측하면서 단어와 문장을 만들어가기 때문에 작은 dataset에서는 철자 오류가 발생할 수 있다.

---

## 22. Additional Generated Text Comparison

추가 실험으로 100 / 500 / 1000 training steps에서 generated text를 비교하였다.

### 22.1 Generated Text After 100 Steps

```text
fllThawit
ote acars ce contautors Uls drm. arld ts uvnss blanaie mancthe or ponthensthesoste heFhjoTSae is fin, hre fBfoe erd ls, wes ditotute bfobinse blinthiledtcle he melvensh wowhe scn. toPe n wlangons ng sionrowoy arotangishoaBhim, arsW.

Imlo The ori ar aceden thilanMqheter ery fory fandyxris
```

100 step에서는 아직 대부분의 출력이 random character sequence에 가깝다.

일부 영어 단어와 짧은 문장 조각이 나타나지만, 전체적인 문장 구조나 의미는 아직 안정적으로 형성되지 않았다.

이 단계에서는 모델이 dataset의 기본적인 character distribution을 학습하기 시작한 수준이라고 볼 수 있다.

---

### 22.2 Generated Text After 500 Steps

```text
Llowt without maiding les caneh becomes more listspoves. The mis not dovidis with conaastyle peFojely ffimilicty. Pe shows sday some revingues dirang eansud cclaite ivewers shos, with suand to chalmices behings otimion ualtihuagus BeiRplaise memplore remsc ineveaud iffele hos forgespcones.

This n
```

500 step에서는 출력이 movie-review-style 문장과 조금 더 비슷해지기 시작한다.

문장 구조는 여전히 불완전하지만, `becomes`, `shows`, `viewers`, `characters`와 비슷한 표현들이 나타난다.

또한 validation loss가 가장 낮아지는 구간이 중간 step 부근이었기 때문에, 이 단계가 generalization 측면에서는 비교적 안정적인 checkpoint라고 볼 수 있다.

---

### 22.3 Generated Text After 1000 Steps

```text
The main comportaraphered ive the characters do not very not als the probleout themselves.

The visual style isthe parts ond a nother and the differenk conversations.

The film shows that memory is not always reliable. Hes .
The film is less interested in providinem the behind twa the characters is
```

1000 step에서는 더 명확하게 movie-review-style 표현이 나타난다.

특히 다음 문장은 dataset의 주제와 문체를 잘 반영한다.

```text
The film shows that memory is not always reliable.
```

이 문장은 영화 리뷰 dataset에서 자주 등장하는 `film`, `memory`, `character`, `visual style` 같은 주제와 연결된다.

다만 여전히 일부 단어가 깨져 있고 문법적으로 부자연스러운 부분이 있다.

이는 모델이 character-level로 학습되었고, dataset 크기도 실제 language model 학습에 비해 매우 작기 때문이다.

---

## 23. How to Run

필요한 package를 설치한다.

```bash
pip install -r requirements.txt
```

프로젝트 검사 코드를 실행한다.

```bash
python check_project.py
```

기본 학습을 실행한다.

```bash
python train.py
```

100 / 500 / 1000 step 비교 실험을 실행한다.

```bash
python main.py
```

학습된 checkpoint로 text generation을 실행한다.

```bash
python generate.py
```

---

## 24. Oral Test Explanation

구술 테스트에서는 전체 구조를 다음 흐름으로 설명할 수 있다.

```text
character encoding
→ token embedding + positional embedding
→ single self-attention head
→ causal mask
→ multi-head attention
→ feedforward network
→ transformer block
→ stacked transformer blocks
→ language modeling head
→ next-character prediction
→ autoregressive generation
```

한국어 설명은 다음과 같다.

```text
먼저 input.txt에 있는 custom dataset을 문자 단위로 읽어서 각 문자를 숫자로 인코딩했습니다.

그 숫자들을 token embedding으로 벡터로 바꾸고, sequence 안에서의 위치 정보를 주기 위해 positional embedding을 더했습니다.

self-attention에서는 입력으로부터 query, key, value를 만들고, query와 key의 내적으로 attention score를 계산했습니다.

GPT는 다음 문자를 예측하는 autoregressive model이기 때문에, 현재 위치가 미래 문자를 보면 안 됩니다. 그래서 causal mask를 적용해서 미래 token에 대한 attention을 막았습니다.

single attention head는 하나의 관점에서 token 관계를 학습합니다. 이러한 head를 여러 개 병렬로 사용하고 concat하면 multi-head attention이 됩니다.

그 다음 multi-head attention과 feedforward network를 하나의 transformer block으로 구성했습니다.

각 block에는 residual connection과 LayerNorm을 사용했습니다. residual connection은 gradient 흐름을 돕고, LayerNorm은 학습을 안정적으로 만듭니다.

이 transformer block을 여러 개 쌓아서 GPT-style decoder-only Transformer를 만들었습니다.

마지막 language modeling head는 각 위치에서 다음 문자의 확률을 출력합니다.

generate 함수에서는 모델이 예측한 다음 문자를 다시 입력 뒤에 붙이고, 이 과정을 반복하면서 새로운 text를 생성합니다.
```

영어 설명은 다음과 같다.

```text
First, I loaded a custom text dataset from input.txt and encoded the text at the character level.

Each character was converted into an integer token. Then, the model used token embedding to convert the token indices into vectors.

I also added positional embedding because the Transformer does not know the order of tokens by itself.

In self-attention, the model creates query, key, and value vectors from the input.

The attention score is computed using the dot product between query and key.

Because GPT is an autoregressive model, each position should not see future tokens. Therefore, I used a causal mask to block attention to future positions.

A single attention head performs this process once. Multiple heads are used in parallel and their outputs are concatenated to make multi-head attention.

Then, multi-head attention and a feedforward network are combined into one transformer block.

Each block uses residual connections and layer normalization.

By stacking multiple transformer blocks, I built a small GPT-style decoder-only Transformer.

Finally, the language modeling head predicts the next character.

During generation, the model repeatedly predicts the next character and appends it to the input sequence.
```

---

## 25. Limitations

이번 프로젝트는 GPT-style model의 핵심 구조를 직접 구현하고 학습 과정을 확인하는 데 목적이 있다.

다만 다음과 같은 한계가 있다.

1. Dataset size가 매우 작다.
2. Character-level model이므로 단어 단위 의미 학습에는 한계가 있다.
3. Validation loss가 후반부에 증가하는 overfitting 현상이 나타났다.
4. Codespaces CPU 환경에서 실행했기 때문에 모델 크기와 학습 step을 제한하였다.
5. 실제 GPT-2 규모의 pretrained language model과 비교할 수 있는 수준은 아니다.

---

## 26. Future Improvements

개선 방향은 다음과 같다.

1. Dataset 크기 확장
2. 더 다양한 주제의 text 추가
3. Word-level 또는 subword-level tokenizer 적용
4. Validation loss 기준 early stopping 추가
5. GPU 환경에서 더 긴 step 학습
6. 더 큰 Transformer model 실험
7. Temperature와 top-k sampling을 조정하여 generation 품질 비교

---

## 27. Conclusion

이 프로젝트에서는 수업에서 학습한 GPT 구조를 바탕으로 작은 character-level decoder-only Transformer를 직접 구현하였다.

Custom movie-review-style dataset을 이용해 모델을 학습하였고, tiny Shakespeare는 사용하지 않았다.

모델은 character-level encoding, token embedding, positional embedding, masked self-attention, multi-head attention, feedforward network, residual connection, LayerNorm, stacked transformer blocks, language modeling head를 포함한다.

최종 학습 결과 training loss는 `3.9340`에서 `0.5865`까지 감소하였다.

Validation loss는 초반과 중반에는 감소했지만, 후반에는 약간 증가하였다. 이는 작은 dataset에서 overfitting이 발생할 수 있음을 보여준다.

학습 후 `checkpoints/tiny_gpt.pt`에 저장된 checkpoint를 불러와 text generation을 수행하였다.

생성 결과는 완벽하지는 않았지만, `movie`, `audience`, `characters`, `memory`, `ending`, `film`, `scene`, `story`와 같은 movie-review-style dataset의 표현을 포함하였다.

결론적으로, 이 프로젝트는 GPT-style language model의 핵심 구조와 학습 과정을 직접 구현하고, custom dataset을 이용해 next-character prediction과 autoregressive text generation이 정상적으로 작동함을 확인한 결과물이다.
