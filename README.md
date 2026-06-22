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
* multi-head attention
* feedforward network
* residual connection
* LayerNorm
* autoregressive text generation
* train / validation loss 비교
* 100 / 500 / 1000 training steps 실험

최종적으로 모델은 custom movie-review-style dataset을 학습하고, 학습 step이 증가함에 따라 generated text가 어떻게 변하는지 비교하였다.

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
    └── tiny_gpt_experiment.pt
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
dataset은 영화의 분위기, 인물, 기억, 갈등, 결말, 감정 등을 설명하는 문장들로 구성하였다.

이 dataset을 선택한 이유는 다음과 같다.

1. tiny Shakespeare가 아닌 별도의 dataset을 사용하기 위함이다.
2. 문장 구조가 비교적 단순하여 작은 GPT 모델이 학습하기 적합하다.
3. `The film`, `The story`, `The character`, `The ending` 등 반복되는 표현이 있어 character-level model이 문체를 학습하는지 확인하기 좋다.
4. 생성 결과에서 영화 리뷰와 비슷한 표현이 나타나는지 비교하기 쉽다.

실행 결과 dataset의 크기는 다음과 같다.

```text
number of characters: 6,666
vocab size: 46
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

모델이 `f` 위치를 예측할 때 그 뒤의 `i`, `l`, `m`을 미리 보면 안 된다.
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

`QK^T`를 `sqrt(d_k)`로 나누는 이유는 dot product 값이 지나치게 커지는 것을 막기 위해서이다. 값이 너무 커지면 softmax가 한쪽으로 과하게 쏠릴 수 있기 때문에 scaling을 적용한다.

---

## 9. Multi-Head Attention

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

---

## 10. Feedforward Network

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

## 11. Residual Connection and LayerNorm

Transformer block에서는 residual connection과 LayerNorm을 사용한다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Residual connection은 기존 입력 `x`를 보존하면서 attention 또는 feedforward의 결과를 더한다.
이를 통해 깊은 모델에서도 gradient가 더 잘 전달될 수 있다.

LayerNorm은 각 token representation의 분포를 정규화하여 학습을 안정적으로 만든다.

이 프로젝트에서는 sublayer를 통과하기 전에 LayerNorm을 먼저 적용하는 pre-norm 구조를 사용하였다.

---

## 12. Output Shape and Loss

모델은 각 위치에서 다음 문자가 vocabulary의 어떤 문자일지 예측한다.

검사 코드 실행 결과는 다음과 같다.

```text
logits shape: torch.Size([64, 46])
loss: 3.9171
```

여기서 `46`은 vocabulary size를 의미한다.
모델의 forward 과정에서는 cross entropy loss 계산을 위해 batch dimension과 sequence dimension을 합친 형태로 logits를 변환한다.

일반적인 형태는 다음과 같다.

```python
B, T, C = logits.shape
logits = logits.view(B * T, C)
targets = targets.view(B * T)
loss = F.cross_entropy(logits, targets)
```

`F.cross_entropy`는 내부적으로 softmax와 negative log likelihood 계산을 포함한다.
따라서 학습 과정에서 logits에 softmax를 미리 적용하지 않는다.

---

## 13. Training Step

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

## 14. Hyperparameters

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

검사 코드에서 확인한 parameter 수는 다음과 같다.

```text
number of parameters: 811,822
```

---

## 15. Project Check

본격적인 학습 전에 `check_project.py`를 작성하여 프로젝트 구성 요소가 정상적으로 작동하는지 확인하였다.

실행 명령어는 다음과 같다.

```bash
python check_project.py
```

실행 결과는 다음과 같다.

```text
GPT2 project check start
------------------------------------------------------------
[1] Config check
batch_size: 32
block_size: 64
n_embd: 128
n_head: 4
n_layer: 4
dropout: 0.2
max_iters: 1000
eval_interval: 100
eval_iters: 100
learning_rate: 0.0003
device: cpu
data_path: data/input.txt
checkpoint_path: checkpoints/tiny_gpt.pt

[2] Dataset check
number of characters: 6,666
vocab size: 46
encoded length: 6,666
train dataset length: 5,935
validation dataset length: 603
x shape: torch.Size([64])
y shape: torch.Size([64])

[3] Model forward check
number of parameters: 811,822
logits shape: torch.Size([64, 46])
loss: 3.9171

[4] Generate check
generated sample:

gFtUv
NicPu.muay
HIdUF:qr::koPlr.M-UrMBcIRMHcrqem,

GPT2 project check completed successfully.
```

이 검사를 통해 다음 요소들이 정상 작동함을 확인하였다.

| Check Item                    | Result |
| ----------------------------- | ------ |
| config import                 | 정상     |
| dataset loading               | 정상     |
| tokenizer encoding / decoding | 정상     |
| train / validation split      | 정상     |
| dataset indexing              | 정상     |
| model creation                | 정상     |
| forward pass                  | 정상     |
| loss calculation              | 정상     |
| generate function             | 정상     |

생성 결과는 아직 학습 전 모델이므로 무작위 문자에 가까운 결과가 나왔다.
이는 정상적인 결과이며, 이후 training을 통해 dataset의 패턴을 학습하게 된다.

---

## 16. Training Experiment: Dataset Expansion and Step Comparison

이번 실험에서는 Tiny GPT 모델을 학습시키면서 `100 steps`, `500 steps`, `1000 steps` 지점에서 loss와 generated text를 비교하였다.

처음에는 `6,666 characters` 크기의 custom movie-review-style dataset을 사용하였다.
하지만 original dataset으로 1000 step까지 학습했을 때, training loss는 계속 감소했지만 validation loss는 오히려 증가하였다.

이는 모델이 작은 dataset을 빠르게 외우면서 overfitting이 발생했음을 의미한다.
따라서 같은 movie-review style을 유지하면서 dataset을 확장하였고, 확장 후 동일한 모델 구조와 학습 설정으로 다시 실험을 진행하였다.

---

### 16.1 Dataset Comparison

| Dataset Version  | Number of Characters | Vocabulary Size |
| ---------------- | -------------------: | --------------: |
| Original Dataset |                6,666 |              46 |
| Expanded Dataset |               11,479 |              48 |

Expanded dataset에는 영화의 분위기, 인물 관계, 기억, 감정, 결말, 시각적 연출 등에 대한 문장을 추가하였다.

Dataset을 확장한 이유는 다음과 같다.

1. 작은 dataset에서 발생한 overfitting을 완화하기 위해서이다.
2. 모델이 더 다양한 sentence pattern을 학습할 수 있도록 하기 위해서이다.
3. movie-review-style의 문체는 유지하면서, 표현과 주제를 더 다양하게 만들기 위해서이다.
4. generated text가 단순 반복이나 memorization에만 의존하지 않는지 확인하기 위해서이다.

---

### 16.2 Original Dataset Training Result

Original dataset을 사용했을 때의 학습 결과는 다음과 같다.

| Training Step | Interval Train Loss | Evaluation Train Loss | Validation Loss |
| ------------: | ------------------: | --------------------: | --------------: |
|           100 |              2.7606 |                2.3951 |          2.4734 |
|           500 |              1.4889 |                0.3527 |          2.9737 |
|          1000 |              0.2196 |                0.1548 |          3.4370 |

Original dataset에서는 training loss가 빠르게 감소하였다.
100 step에서 evaluation train loss는 `2.3951`이었고, 1000 step에서는 `0.1548`까지 감소하였다.

하지만 validation loss는 100 step 이후 계속 증가하였다.
100 step에서 validation loss는 `2.4734`였지만, 500 step에서는 `2.9737`, 1000 step에서는 `3.4370`까지 증가하였다.

즉, 모델이 training dataset에는 점점 더 잘 맞춰졌지만, validation dataset에 대해서는 성능이 나빠졌다.
이는 original dataset의 크기가 작아서 모델이 data pattern을 일반화하기보다는 training text를 빠르게 memorization했기 때문으로 볼 수 있다.

---

### 16.3 Expanded Dataset Training Result

Dataset을 `11,479 characters`로 확장한 뒤 같은 설정으로 다시 학습하였다.

Expanded dataset을 사용했을 때의 학습 결과는 다음과 같다.

| Training Step | Interval Train Loss | Evaluation Train Loss | Validation Loss |
| ------------: | ------------------: | --------------------: | --------------: |
|           100 |              2.7436 |                2.3953 |          2.3669 |
|           500 |              1.6849 |                0.6583 |          1.8990 |
|          1000 |              0.3174 |                0.1980 |          2.4929 |

Expanded dataset에서는 validation loss가 original dataset보다 전반적으로 낮아졌다.

특히 500 step에서 validation loss는 original dataset의 `2.9737`에서 expanded dataset의 `1.8990`으로 크게 개선되었다.
1000 step에서도 original dataset의 validation loss는 `3.4370`이었지만, expanded dataset에서는 `2.4929`로 더 낮게 유지되었다.

또한 expanded dataset에서는 training loss가 감소하긴 했지만, original dataset처럼 지나치게 빠르게 0에 가까워지지는 않았다.
이는 dataset이 확장되면서 모델이 단순히 training text를 외우는 정도가 줄어들고, 더 다양한 character pattern을 학습하게 되었기 때문으로 해석할 수 있다.

---

### 16.4 Comparison of Validation Loss

두 dataset의 validation loss를 비교하면 다음과 같다.

| Training Step | Original Dataset Val Loss | Expanded Dataset Val Loss |   Change |
| ------------: | ------------------------: | ------------------------: | -------: |
|           100 |                    2.4734 |                    2.3669 | Improved |
|           500 |                    2.9737 |                    1.8990 | Improved |
|          1000 |                    3.4370 |                    2.4929 | Improved |

모든 checkpoint에서 expanded dataset의 validation loss가 더 낮게 나타났다.

이 결과는 dataset 확장이 모델의 generalization에 도움이 되었음을 보여준다.
특히 500 step에서 validation loss가 가장 낮았기 때문에, 현재 실험에서는 expanded dataset 기준으로 500 step 부근이 가장 안정적인 학습 지점이라고 볼 수 있다.

1000 step에서는 generated text가 더 길고 dataset의 문체를 더 많이 반영하지만, validation loss는 500 step보다 증가하였다.
따라서 1000 step은 training text에 대한 학습은 더 많이 진행되었지만, generalization 측면에서는 500 step보다 덜 안정적이라고 해석할 수 있다.

---

### 16.5 Generated Text with Expanded Dataset

Expanded dataset으로 학습한 모델의 generated text는 다음과 같다.

#### Generated Text After 100 Steps

```text
fllThawit
ote acars ce contautors Uls drm. arld ts uvnss blanaie mancthe or ponthensthesoste heFhjoTSae is fin, hre fBfoe erd ls, wes ditotute bfobinse blinthiledtcle he melvensh wowhe scn. toPe n wlangons ng sionrowoy arotangishoaBhim, arsW.

Imlo The ori ar aceden thilanMqheter ery fory fandyxris
```

100 step에서는 아직 대부분의 출력이 random character sequence에 가깝다.
일부 영어 단어와 짧은 문장 조각이 나타나지만, 전체적인 문장 구조나 의미는 아직 안정적으로 형성되지 않았다.

이 단계에서는 모델이 dataset의 기본적인 character distribution을 학습하기 시작한 수준이라고 볼 수 있다.

---

#### Generated Text After 500 Steps

```text
Llowt without maiding les caneh becomes more listspoves. The mis not dovidis with conaastyle peFojely ffimilicty. Pe shows sday some revingues dirang eansud cclaite ivewers shos, with suand to chalmices behings otimion ualtihuagus BeiRplaise memplore remsc ineveaud iffele hos forgespcones.

This n
```

500 step에서는 출력이 movie-review-style 문장과 조금 더 비슷해지기 시작한다.
문장 구조는 여전히 불완전하지만, `becomes`, `shows`, `viewers`, `characters`와 비슷한 표현들이 나타난다.

또한 expanded dataset 기준으로 validation loss가 가장 낮은 시점이 500 step이었기 때문에, 이 단계가 현재 실험에서는 가장 안정적인 checkpoint라고 볼 수 있다.

---

#### Generated Text After 1000 Steps

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

### 16.6 Interpretation

이번 실험을 통해 dataset size가 GPT-style language model의 학습 결과에 큰 영향을 준다는 것을 확인하였다.

Original dataset에서는 training loss가 빠르게 감소했지만 validation loss가 크게 증가하였다.
이는 모델이 작은 dataset을 빠르게 외우면서 overfitting이 발생했음을 의미한다.

반면 expanded dataset에서는 validation loss가 전반적으로 낮아졌고, generated text도 더 자연스러운 movie-review-style 문장 구조를 보였다.
특히 500 step에서 validation loss가 가장 낮았기 때문에, 현재 실험에서는 500 step 부근이 가장 적절한 학습 지점으로 판단된다.

결론적으로 dataset을 확장한 후 모델은 더 다양한 문장 패턴을 학습할 수 있었고, original dataset에 비해 generalization이 개선되었다.

이번 실험은 단순히 모델이 학습되는지만 확인한 것이 아니라, dataset size와 training step이 loss 및 generated text quality에 어떤 영향을 주는지 비교했다는 점에서 의미가 있다.


---

## 17. Generated Text Comparison

### 17.1 Generated Text After 100 Steps

```text
jbthe kicanMFe turacond Th imaEinduy othavene in.t q, tInou tilcast inThatsie ze dos hee ter stit mSoxpoqhi,tlmariuthe ot cpts bezifeare ot tthe h

, verefinMira fira thzes walelvesir.FostoutMOndxt wjols itt acoij kporuse Faramol tathe ThhieI she serangivhaltul, cs ssdeve he bud ald sFouond ospe fh
```

100 step에서는 아직 대부분의 출력이 random character sequence에 가깝다.
일부 영어 단어처럼 보이는 조각이 나타나지만, 문장 구조나 의미는 거의 형성되지 않았다.

이 단계에서는 모델이 아직 dataset의 기본적인 character distribution만 조금씩 학습하는 수준이라고 볼 수 있다.

---

### 17.2 Generated Text After 500 Steps

```text
The film shows that reces inflict.

The flie s not chomItarapin beher tsafole besel, bet the stisembe spome, than sthe diffuterenz fore. Thie s
Thoma charactMuse in nother a notlinue. Fot byeMus atorjus iest nsot akeRabid unsurie t
altor, afocehe selvion. The scresis tarens a ta stak peoss possible
```

500 step에서는 출력이 movie-review dataset의 문체를 조금 더 따라가기 시작했다.
예를 들어 `The film shows`와 같은 표현이 등장하고, 전체적으로 영화 리뷰 문장처럼 시작하는 구조가 나타난다.

다만 단어 단위에서는 아직 깨진 표현이 많다.
이는 character-level model이기 때문에 단어를 직접 학습하는 것이 아니라, 문자 단위의 다음 글자를 예측하면서 문장을 만들어가기 때문이다.

---

### 17.3 Generated Text After 1000 Steps

```text
One film austic unreaeral fte mor the film reveals the Ince isprons hee lfrienct. Itaind senchoir mating pa dis rfothe bradinon the story continues. At firstMure itses noto c pertension. at film focul feaul wm beaver with rathe lmthe samore. The ais mevoing of eryst focuses. The most inot abander h
```

1000 step에서는 더 긴 문장 구조가 나타나고, `the film`, `the story continues`, `focuses`와 같이 movie-review dataset에서 자주 등장할 법한 표현들이 생성되었다.

즉, 모델이 dataset의 분위기와 반복되는 표현을 어느 정도 학습했다는 것을 확인할 수 있다.

하지만 여전히 많은 단어가 불완전하거나 철자가 깨져 있다.
이는 dataset의 크기가 작고, 모델이 word-level이 아니라 character-level로 학습되었기 때문이다.

---

## 18. Result Interpretation

이번 실험을 통해 두 가지를 확인할 수 있었다.

첫째, Tiny GPT 모델은 정상적으로 학습되었다.
Training loss가 계속 감소했고, generated text도 100 step에서는 거의 무작위였지만 500 step과 1000 step에서는 영화 리뷰와 비슷한 표현을 포함하기 시작했다.

둘째, 작은 custom dataset에서는 overfitting이 빠르게 발생한다.
Training loss는 감소했지만 validation loss는 증가했기 때문에, 모델이 training text에는 점점 더 잘 맞춰졌지만 validation text에 대한 일반화 성능은 떨어졌다.

따라서 이번 프로젝트는 GPT 구조와 training pipeline이 정상적으로 구현되었음을 보여주는 동시에, dataset size가 모델 성능과 generalization에 큰 영향을 준다는 점도 확인할 수 있었다.

---

## 19. How to Run

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

## 20. Limitations

이번 프로젝트는 GPT-style model의 핵심 구조를 직접 구현하고 학습 과정을 확인하는 데 목적이 있다.

다만 다음과 같은 한계가 있다.

1. Dataset size가 매우 작다.
2. Character-level model이므로 단어 단위 의미 학습에는 한계가 있다.
3. Validation loss가 증가하는 overfitting 현상이 나타났다.
4. Codespaces CPU 환경에서 실행했기 때문에 모델 크기와 학습 step을 제한하였다.
5. 실제 GPT-2 규모의 pretrained language model과 비교할 수 있는 수준은 아니다.

---

## 21. Future Improvements

개선 방향은 다음과 같다.

1. Dataset 크기 확장
2. 더 다양한 주제의 text 추가
3. Word-level 또는 subword-level tokenizer 적용
4. Validation loss 기준 early stopping 추가
5. GPU 환경에서 더 긴 step 학습
6. 더 큰 Transformer model 실험
7. Temperature와 top-k sampling을 조정하여 generation 품질 비교

---

## 22. Conclusion

이 프로젝트에서는 수업에서 학습한 GPT 구조를 바탕으로 작은 character-level decoder-only Transformer를 직접 구현하였다.

Custom movie-review dataset을 이용해 모델을 학습하였고, 100 / 500 / 1000 step에서 loss와 generated text를 비교하였다.

학습 결과 training loss는 꾸준히 감소했으며, generated text도 학습이 진행될수록 dataset의 문체를 더 많이 반영하였다.
반면 validation loss는 증가하여 작은 dataset에서 overfitting이 발생한다는 점도 확인할 수 있었다.

결론적으로, 이 프로젝트는 GPT-style language model의 핵심 구조와 학습 과정을 직접 구현하고 실험한 결과물이다.
