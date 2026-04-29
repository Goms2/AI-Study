# 📘 Attention Is All You Need (Transformer) 논문 정리

> **Vaswani et al., 2017** | NIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
>
> Transformer 모델을 세상에 처음 선보인 **전설적인 논문** ⚡

---

## 🗂️ 목차

1. [Abstract (3줄 요약)](#-abstract-3줄-요약)
2. [핵심 문제 & 해결책](#-핵심-문제--해결책)
3. [핵심 구조 설명](#-핵심-구조-설명)
4. [Transformer 전체 구조](#-transformer-전체-구조)
5. [논문 ↔ 코드 연결 (PyTorch)](#-논문--코드-연결-pytorch)
6. [전체 모델 구현 코드](#-전체-모델-구현-코드)
7. [데이터 흐름 추적 (Shape 변화)](#-데이터-흐름-추적-shape-변화)

---

## ✏️ Abstract (3줄 요약)

| 구분 | 내용 |
|------|------|
| 🔴 **문제** | 기존 RNN/CNN 기반 번역 모델은 **단어를 순서대로 처리**해야 해서 학습이 느리고, 문장이 길어지면 **장거리 의존성** 학습이 어려움 |
| 🟡 **방법** | 순환(Recurrence)과 합성곱(Convolution)을 모두 제거하고, **오직 Attention 메커니즘만으로** 구성된 **Transformer** 제안 |
| 🟢 **결과** | WMT 2014 영→독 **BLEU 28.4**, 영→불 **BLEU 41.8** 달성. 학습 시간도 기존 대비 대폭 단축 (8 GPU × 3.5일) |

---

## 🔍 핵심 문제 & 해결책

### 기존 RNN 방식의 문제점

#### 1. 순차 계산 → 병렬화 불가능
> RNN은 단어를 **하나씩 순서대로** 처리합니다.
> `"I love you"` → `I` → `love` → `you` 순서로만 처리 가능
>
> **→ GPU를 여러 개 써도 속도가 안 올라감** 😵

#### 2. 장거리 의존성 학습 어려움
> 문장이 길어지면 앞쪽 단어 정보가 뒤로 갈수록 **희미해집니다.**
>
> 예: `"내가 어제 도서관에서 빌린 그 책은 재미있다"`
> → `"책"`과 `"재미있다"`의 연결이 약해짐

#### 3. CNN 기반 모델도 한계
> ConvS2S, ByteNet도 멀리 있는 단어를 연결하려면 층을 많이 쌓아야 함

### Transformer의 해결책

> 💡 **핵심 아이디어**: RNN/CNN을 완전히 버리고, **Attention만으로** 모든 걸 해결하자!

```
✅ 모든 단어를 한 번에 병렬 처리       → 학습 속도 대폭 향상
✅ 모든 단어 쌍 관계를 한 번에 계산    → 장거리 의존성 해결
✅ 번역 품질도 SOTA 달성              → 효율 + 성능 모두 잡음
```

---

## 🧱 핵심 구조 설명

Transformer는 **Encoder(6층)** + **Decoder(6층)** 으로 구성되어 있습니다.

---

### ① Input Embedding + Positional Encoding ⭐

> 🎓 **비유**: 학생들(단어)에게 **출석번호(위치)** 를 달아주는 것

컴퓨터는 글자를 모르므로, 단어를 숫자 벡터로 바꿔줘야 합니다 (Embedding).
그런데 Transformer는 단어를 한꺼번에 받기 때문에, **순서 정보가 사라집니다.**
그래서 각 단어에 위치 정보를 더해줍니다 (Positional Encoding).

```
"나는 너를 좋아해"  vs  "너를 나는 좋아해"
→ 위치 정보가 없으면 둘을 구분 못 함!
```

**Positional Encoding 공식 (sin/cos 사용):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, seq_len)` — 토큰 ID |
| **출력** | `(batch, seq_len, d_model=512)` |
| **하이퍼파라미터** | `d_model=512`, `vocab_size≈37,000` |

---

### ② Self-Attention ⭐

> 📚 **비유**: **도서관에서 책 찾기**
> - **Query**: "사랑에 관한 책이 필요해" (질문)
> - **Key**: 도서관의 모든 책 제목 목록
> - **Value**: 각 책의 실제 내용
>
> → Query와 Key를 비교해 **관련도 점수**를 매기고, 점수대로 Value를 섞어 가져옴

**수식:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**왜 √d_k로 나눌까?**
> Q·K 내적값이 차원이 클수록 커져서 softmax가 한쪽에 극단적으로 쏠립니다 (기울기 소실).
> √d_k로 나눠 안정화합니다.

| 항목 | 내용 |
|------|------|
| **입력** | Q, K, V — 각 `(batch, seq_len, d_k)` |
| **출력** | `(batch, seq_len, d_v)` |
| **하이퍼파라미터** | `d_k=64`, `d_v=64` |

---

### ③ Multi-Head Attention ⭐

> 👥 **비유**: 한 명의 독자보다 **여러 전문가**가 각자 다른 관점에서 책을 읽으면 더 풍부한 이해
> - 전문가 1: 문법 관점
> - 전문가 2: 감정 관점
> - 전문가 3: 시간 순서 관점
> - ...

Attention을 **8번 병렬로 수행**하고 결과를 합칩니다.

**작동 방식:**
```
1. Q, K, V를 8개로 쪼갬 (512 → 64 × 8)
2. 각 head가 독립적으로 Attention 계산
3. 결과 8개를 이어 붙임 (concat) → 512차원 복원
4. 마지막에 Linear 투영
```

| 항목 | 내용 |
|------|------|
| **입력** | Q, K, V — `(batch, seq_len, d_model=512)` |
| **출력** | `(batch, seq_len, d_model=512)` |
| **하이퍼파라미터** | `h=8` (head 개수), `d_k = d_v = 512/8 = 64` |

---

### ④ Masked Multi-Head Attention (Decoder 전용) ⭐

> 📝 **비유**: **시험 볼 때 뒷장 미리 보면 안 됨**
>
> 디코더는 왼쪽부터 한 단어씩 생성해야 하므로, **미래 단어를 보면 안 됩니다.**

```
번역 생성 과정 (예: "I love you" → "나는 너를 사랑해")
- "나는"을 만들 때: 아직 "너를", "사랑해"를 모르는 상태여야 함!
```

**구현 방법**: 미래 단어 위치의 attention 점수를 `-∞`로 막아 → softmax 후 0이 되게 함

| 항목 | 내용 |
|------|------|
| **입력** | Q, K, V + **mask 행렬** |
| **출력** | `(batch, seq_len, d_model)` |
| **하이퍼파라미터** | Multi-Head Attention과 동일 |

---

### ⑤ Position-wise Feed-Forward Network (FFN)

> 🔧 **비유**: 각 단어마다 독립적으로 적용되는 **작은 MLP**
>
> Attention은 "단어들 사이의 관계"를, FFN은 "각 단어 자체의 특징"을 깊게 가공

**수식:**

```
FFN(x) = max(0, xW₁ + b₁) W₂ + b₂
       = Linear → ReLU → Linear
```

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, seq_len, d_model=512)` |
| **출력** | `(batch, seq_len, d_model=512)` |
| **하이퍼파라미터** | `d_ff=2048` (중간 차원: 512 → 2048 → 512) |

---

### ⑥ Residual Connection + Layer Normalization

> 🔁 **비유**: **"원본 복사본 남기기"** (ResNet 아이디어)
>
> 복잡한 처리를 거쳐도 원본 정보가 사라지지 않도록 입력을 그대로 더해줌

**수식:**

```
output = LayerNorm(x + Sublayer(x))
```

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, seq_len, d_model)` |
| **출력** | `(batch, seq_len, d_model)` (shape 동일) |
| **하이퍼파라미터** | 없음 (학습 파라미터만 존재) |

참고: 논문 원본에서는 Sublayer(x) + x를 먼저 하고 그 결과에 LayerNorm을 적용합니다. 하지만 최근 PyTorch 공식 구현이나 성능 최적화 모델들은 LayerNorm을 먼저 하는 Pre-LN 구조를 더 많이 사용

> 논문 방식 (Post-LN): LayerNorm(x + Sublayer(x))

> 최근 트렌드 (Pre-LN): x + Sublayer(LayerNorm(x))

---

### ⑦ Encoder-Decoder Attention (Decoder 2번째 sub-layer)

> 🌉 **비유**: 번역할 때 **"원문의 어떤 단어에 집중할까?"** 를 결정하는 다리

| 출처 | 역할 |
|------|------|
| **Query** | Decoder의 이전 층 출력 (지금 만드는 단어) |
| **Key** | Encoder의 최종 출력 (원문 전체) |
| **Value** | Encoder의 최종 출력 (원문 전체) |

| 항목 | 내용 |
|------|------|
| **입력** | Q는 Decoder, K/V는 Encoder에서 |
| **출력** | `(batch, tgt_len, d_model)` |
| **하이퍼파라미터** | Multi-Head Attention과 동일 |

---

## 🏗️ Transformer 전체 구조

```
            [Inputs]                                [Outputs (shifted right)]
               │                                              │
       Input Embedding                              Output Embedding
               │                                              │
       + Positional Encoding                        + Positional Encoding
               │                                              │
   ┌───────────▼────────────┐               ┌─────────────────▼─────────────┐
   │  Multi-Head            │               │  Masked Multi-Head            │
   │  Self-Attention        │               │  Self-Attention               │
   └───────────┬────────────┘               └─────────────────┬─────────────┘
         Add & Norm                                     Add & Norm
   ┌───────────▼────────────┐               ┌─────────────────▼─────────────┐
   │   Feed Forward         │               │  Encoder-Decoder Attention    │◄── (K, V)
   └───────────┬────────────┘               │  (Q는 Decoder, K/V는 Encoder) │
         Add & Norm                         └─────────────────┬─────────────┘
               │                                        Add & Norm
       × N=6 layers                          ┌─────────────────▼─────────────┐
               │                             │       Feed Forward            │
            [Encoder]    ──────────►         └─────────────────┬─────────────┘
         (출력을 K, V로 전달)                              Add & Norm
                                                               │
                                                        × N=6 layers
                                                               │
                                                        Linear + Softmax
                                                               │
                                                    [Output Probabilities]
                                                          [Decoder]
```

### 데이터 흐름 요약

1. **입력 문장** → 토큰화 → Embedding + Positional Encoding → `(batch, seq_len, 512)`
2. **Encoder (6층 반복)**: Self-Attention → FFN (각 층마다 Residual + LayerNorm)
3. Encoder 최종 출력(K, V)을 **Decoder로 전달**
4. **Decoder (6층 반복)**: Masked Self-Attention → Encoder-Decoder Attention → FFN
5. 최종 **Linear + Softmax** → 다음 단어의 확률 분포 출력

---

## 💻 논문 ↔ 코드 연결 (PyTorch)

| 논문 표현 | PyTorch 코드 |
|-----------|-------------|
| `N = 6 identical layers` | `nn.ModuleList([EncoderLayer() for _ in range(6)])` |
| `d_model = 512` | `d_model = 512` |
| `softmax(QK^T / √d_k) V` | `torch.softmax(Q @ K.transpose(-2,-1) / math.sqrt(d_k), dim=-1) @ V` |
| `h = 8 parallel attention layers` | `num_heads = 8`, `d_k = 512 // 8 = 64` |
| `FFN(x) = max(0, xW₁+b₁)W₂+b₂` | `nn.Sequential(nn.Linear(512,2048), nn.ReLU(), nn.Linear(2048,512))` |
| `residual + layer normalization` | `x = self.norm(x + self.sublayer(x))` |
| `d_ff = 2048` | `d_ff = 2048` |
| `Masked self-attention (미래 단어 차단)` | `scores.masked_fill(mask == 0, -1e9)` |
| `dropout rate P_drop = 0.1` | `nn.Dropout(0.1)` |
| `label smoothing ε_ls = 0.1` | `nn.CrossEntropyLoss(label_smoothing=0.1)` |

---

## 🖥️ 전체 모델 구현 코드

```python
import torch
import torch.nn as nn
import math


# 1. Scaled Dot-Product Attention
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # QK^T / √d_k : 유사도 점수 계산
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 마스킹
    attn = torch.softmax(scores, dim=-1)              # 확률로 변환
    return torch.matmul(attn, V)                       # Value와 가중합


# 2. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_k = d_model // num_heads  # 64
        self.h = num_heads               # 8
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        # 1) Linear projection + 8개 head로 쪼개기
        Q = self.W_q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Attention 수행
        out = scaled_dot_product_attention(Q, K, V, mask)
        # 3) Concat (head 합치기)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # 4) 마지막 Linear
        return self.W_o(out)


# 3. Position-wise Feed-Forward Network
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


# 4. Positional Encoding (sin/cos)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원: cos
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)]


# 5. Encoder Layer (Self-Attn + FFN + Residual + Norm)
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Residual Connection + LayerNorm
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
```

---

## 📊 데이터 흐름 추적 (Shape 변화)

> **예시 설정**: `batch=2`, `seq_len=10`, `d_model=512`, `heads=8`, `d_k=64`

```
입력 토큰 ID
(2, 10)
    │
    ▼  [Embedding]                    각 토큰 ID를 512차원 벡터로
(2, 10, 512)
    │
    ▼  [+ Positional Encoding]        위치 정보 더하기 (shape 유지)
(2, 10, 512)
    │
    ▼  [Linear W_q, W_k, W_v]         Q, K, V 생성
(2, 10, 512) × 3
    │
    ▼  [view + transpose]             8개 head로 쪼개기
(2, 8, 10, 64)                        # batch=2, heads=8, seq=10, d_k=64
    │
    ▼  [Q @ K^T]                      어텐션 점수 행렬
(2, 8, 10, 10)                        # 모든 단어 쌍의 유사도
    │
    ▼  [÷ √64 + softmax]              정규화 + 확률화
(2, 8, 10, 10)                        # attention weights
    │
    ▼  [@ V]                          Value와 가중합
(2, 8, 10, 64)
    │
    ▼  [concat heads]                 8개 head 이어 붙이기
(2, 10, 512)
    │
    ▼  [W_o Linear]                   최종 출력 투영
(2, 10, 512)
    │
    ▼  [+ residual, LayerNorm]        잔차 연결 + 정규화
(2, 10, 512)
    │
    ▼  [Feed Forward: 512→2048→512]
(2, 10, 512)
    │
    ▼  [× 6 layers 반복]
(2, 10, 512)                          # Encoder 최종 출력
    │
    ▼  [Decoder 통과 + Linear]
(2, 10, vocab_size)                   # 각 위치별 다음 단어 확률
```

### 💡 단계별 핵심 포인트

| 변환 | 의미 |
|------|------|
| `(2,10)` → `(2,10,512)` | 토큰 ID를 의미 있는 연속 벡터로 변환 |
| `(2,10,512)` → `(2,8,10,64)` | 여러 관점(head)에서 동시에 attention 계산 |
| `(2,8,10,10)` | **모든 단어 쌍의 관계 점수 — 핵심!** |
| `(2,8,10,64)` → `(2,10,512)` | 8개 head 결과 다시 합침 |
| `(2,10,512)` → `(2,10,vocab_size)` | 최종 다음 단어 예측 확률 |

---

## 📚 참고 자료

- 📄 [원문 논문 (arXiv)](https://arxiv.org/abs/1706.03762)
- 🤗 [Hugging Face Transformers](https://github.com/huggingface/transformers)
- 📖 [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/)
- 🎨 [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

---

<div align="center">

**Attention Is All You Need** ⚡

*"단어를 순서대로 처리할 필요 없다 — 한 번에, 모두, 동시에 보자"*

</div>
