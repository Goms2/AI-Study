# 📄 Deep Residual Learning for Image Recognition (ResNet) 논문 정리

> **He et al., 2015** | CVPR 2016 Best Paper Award | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

---

## 🗂️ 목차

1. [Abstract (3줄 요약)](#-abstract-3줄-요약)
2. [핵심 문제 & 해결책](#-핵심-문제--해결책)
3. [핵심 구조 설명](#-핵심-구조-설명)
4. [잔차 블록 (Residual Block)](#-잔차-블록-residual-block--핵심)
5. [논문 ↔ 코드 연결 (PyTorch)](#-논문--코드-연결-pytorch)
6. [ResNet 구조 한눈에 보기](#-resnet-구조-한눈에-보기)
7. [전체 ResNet-50 구현 코드](#-전체-resnet-50-구현-코드)
8. [데이터 흐름 추적 (Shape 변화)](#-데이터-흐름-추적-shape-변화)

---

## ✏️ Abstract (3줄 요약)

| 구분 | 내용 |
|------|------|
| 🔴 **문제** | 신경망을 깊게 쌓을수록 오히려 학습이 안 되는 **성능 저하(degradation)** 현상 발생 |
| 🟡 **방법** | 정답을 직접 학습하는 대신, **"입력값과의 차이(잔차, residual)"** 만 학습하도록 구조를 변경 + **숏컷 연결(shortcut connection)** 추가 |
| 🟢 **결과** | 152층 초심층 네트워크를 성공적으로 학습 → ImageNet 오류율 **3.57%** 로 1위 달성 |

---

## 🔍 핵심 문제 & 해결책

### 기존 방식의 문제점

딥러닝에서 **"층이 깊을수록 좋다"** 는 건 상식이었지만, 두 가지 큰 문제가 있었습니다.

#### 1. 기울기 소실 (Vanishing Gradient)
> 학습 신호가 앞쪽 층으로 전달되는 과정에서 점점 약해지는 현상.
> 마치 **"전화기 게임"** 처럼, 메시지가 층을 거칠수록 흐릿해집니다.

→ **Batch Normalization** 으로 어느 정도 해결됨

#### 2. 성능 저하 (Degradation) 🔴
> 층을 더 쌓으면 오히려 **훈련 오류(training error)가 높아지는** 현상.
> 과적합(overfitting)이 아닌, **학습 자체가 잘 안 되는** 문제.

```
Figure 1: 56층 네트워크 < 20층 네트워크 (더 깊은데 성능이 더 나쁨!)
```

### ResNet의 해결책

> 💡 **핵심 아이디어**: 층이 정답 `H(x)`를 직접 학습하는 대신,  
> **"원래 입력과 정답의 차이(잔차)"** `F(x) = H(x) - x` 만 학습하게 만들자!

```
기존 방식:  출력 = F(x)          → 처음부터 정답을 학습
ResNet  :  출력 = F(x) + x      → 입력에 수정분만 더함
```

이때 `x`를 그대로 더해주는 연결 = **숏컷 연결(shortcut connection) / 스킵 연결(skip connection)**

---

## 🧱 핵심 구조 설명

### ① 합성곱 층 (Convolutional Layer)

> 🔎 **비유**: 사진에서 특정 패턴(가장자리, 질감, 색상)을 찾아내는 **"돋보기"**

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, C, H, W)` — 예: `(32, 3, 224, 224)` |
| **출력** | `(batch, 필터수, 새H, 새W)` |
| **하이퍼파라미터** | 필터 크기(예: 3×3, 7×7), 필터 개수(채널 수), 스트라이드(stride), 패딩(padding) |

---

### ② 배치 정규화 (Batch Normalization)

> 📊 **비유**: 과목마다 들쭉날쭉한 시험 점수를 **"평균 0, 표준편차 1"** 로 맞춰주는 표준화

| 항목 | 내용 |
|------|------|
| **입력** | 합성곱 층의 출력 텐서 |
| **출력** | 평균 0, 표준편차 1에 가깝게 정규화된 텐서 (shape 동일) |
| **하이퍼파라미터** | 없음 (`γ`, `β`는 학습되는 파라미터) |

---

### ③ ReLU 활성화 함수

> 🚪 **비유**: **"0 이하는 무시, 0 이상은 통과"** 시키는 필터

```python
ReLU(x) = max(0, x)
```

| 항목 | 내용 |
|------|------|
| **입력** | 어떤 shape의 텐서든 가능 |
| **출력** | 같은 shape, 음수 값만 0으로 변경 |
| **하이퍼파라미터** | 없음 |

---

### ④ 전역 평균 풀링 (Global Average Pooling)

> 📊 **비유**: 반 전체 학생 점수를 **"평균 하나"** 로 요약 (H×W 크기의 특징 맵을 픽셀 하나로 압축)

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, C, H, W)` |
| **출력** | `(batch, C, 1, 1) → 보통 (batch, C)로 flatten` |
| **하이퍼파라미터** | 없음 |

---

### ⑤ 완전연결층 + Softmax (FC + Softmax)

> ⚖️ **비유**: 단서를 취합해 **"고양이 80%, 강아지 15%, 새 5%"** 처럼 최종 분류를 내리는 **"판사"**

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, 2048)` |
| **출력** | `(batch, 1000)` — 1000개 클래스 각각의 확률값 |
| **하이퍼파라미터** | 출력 클래스 수 (ImageNet = 1000) |

---

## 🔴 잔차 블록 (Residual Block) — 핵심

> 📝 **비유**: 학생이 답안을 처음부터 쓰는 게 아니라, **"기존 답안에서 틀린 부분만 수정"** 하는 방식

> 기존 방식: y = F(x) → 처음부터 정답을 학습</br>
> ResNet 방식: y = F(x) + x → x(원래 입력)에 수정분만 더함</br>
> 이때 x를 그대로 더해주는 연결이 **숏컷 연결(shortcut connection)** </br>

### Basic Block (ResNet-18 / 34)

```
입력 x
  │
  ├───────────────────────────┐  (숏컷 연결)
  │                           │
Conv 3×3 → BN → ReLU          │
  │                           │
Conv 3×3 → BN                 │
  │                           │
  └───────── (+) ─────────────┘
              │
            ReLU
              │
           출력 y
```

### Bottleneck Block (ResNet-50 / 101 / 152) ⭐

```
입력 x
  │
  ├──────────────────────────────┐  (숏컷 연결)
  │                              │
Conv 1×1 → BN → ReLU  (채널 축소) │
  │                              │
Conv 3×3 → BN → ReLU  (특징 추출) │
  │                              │
Conv 1×1 → BN         (채널 복원) │
  │                              │
  └──────────── (+) ─────────────┘
                 │
               ReLU
                 │
              출력 y
```

> 💡 **1×1 Conv를 쓰는 이유**: 채널을 먼저 줄여(병목, bottleneck) 계산량을 줄이고, 처리 후 다시 채널을 늘림

| 항목 | 내용 |
|------|------|
| **입력** | `(batch, C, H, W)` |
| **출력** | `(batch, C, H, W)` — shape 동일 |
| **채널이 달라지는 경우** | 숏컷도 `1×1 Conv` 로 채널 조정 |
| **하이퍼파라미터** | 블록 반복 횟수(n), 필터 수 |

---

## 💻 논문 ↔ 코드 연결 (PyTorch)

| 논문 표현 | PyTorch 코드 |
|-----------|-------------|
| `7×7 conv, 64 filters, stride 2` | `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)` |
| `BN right after each convolution` | `nn.BatchNorm2d(64)` |
| `ReLU activation` | `nn.ReLU(inplace=True)` |
| `3×3 max pool, stride 2` | `nn.MaxPool2d(kernel_size=3, stride=2, padding=1)` |
| `F(x) + x` (shortcut connection) | `out += self.shortcut(x)`(잔차 덧셈)|
| `1×1 conv to match dimensions` | `nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)` |
| `global average pooling` | `nn.AdaptiveAvgPool2d((1, 1))` |
| `1000-way FC with softmax` | `nn.Linear(2048, 1000)` |

### 전체 Bottleneck Block 코드 예시

```python
import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 1×1 Conv (채널 축소)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 3×3 Conv (특징 추출)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # 1×1 Conv (채널 복원)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # 숏컷 연결 (채널/크기가 다를 경우 조정)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # 1×1
        out = self.relu(self.bn2(self.conv2(out))) # 3×3
        out = self.bn3(self.conv3(out))            # 1×1

        out += self.shortcut(x)  # ✅ 잔차 덧셈 (핵심!)
        out = self.relu(out)
        return out
```

---

## 📐 ResNet 구조 한눈에 보기

```
입력 이미지 (224×224×3)
        │
   7×7 Conv, 64 filters, stride 2
   BatchNorm → ReLU
   3×3 MaxPool, stride 2
        │
   ┌────┴────┐
   │  Layer2 │  Residual Blocks × N
   └────┬────┘
   ┌────┴────┐
   │  Layer3 │  Residual Blocks × N (stride 2로 해상도 절반)
   └────┬────┘
   ┌────┴────┐
   │  Layer4 │  Residual Blocks × N (stride 2로 해상도 절반)
   └────┬────┘
   ┌────┴────┐
   │  Layer5 │  Residual Blocks × N (stride 2로 해상도 절반)
   └────┬────┘
        │
   Global Average Pooling
        │
   FC Layer → Softmax
        │
   출력 (1000개 클래스 확률)
```

| 모델 | 블록 타입 | 층 수 | 파라미터 수 |
|------|-----------|-------|------------|
| ResNet-18 | Basic Block | 18 | 11.7M |
| ResNet-34 | Basic Block | 34 | 21.8M |
| ResNet-50 | Bottleneck | 50 | 25.6M |
| ResNet-101 | Bottleneck | 101 | 44.5M |
| ResNet-152 | Bottleneck | 152 | 60.2M |

---

## 🖥️ 전체 ResNet-50 구현 코드

```python
import torch
import torch.nn as nn

# ① 병목 블록 (Bottleneck Block) - ResNet-50/101/152용
class BottleneckBlock(nn.Module):
    expansion = 4  # 출력 채널 = 입력 채널 × 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()

        # 논문의 F(x) 부분 (잔차 학습)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu  = nn.ReLU(inplace=True)

        # 숏컷 연결 - 채널 수나 크기가 다를 때 1×1 conv로 맞춤
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )

    def forward(self, x):
        identity = x  # 원래 입력값 저장 (숏컷용)

        out = self.relu(self.bn1(self.conv1(x)))   # 1×1 conv
        out = self.relu(self.bn2(self.conv2(out))) # 3×3 conv
        out = self.bn3(self.conv3(out))            # 1×1 conv (ReLU 전)

        # 핵심! F(x) + x
        out += self.shortcut(identity)
        out = self.relu(out)
        return out


# ② 전체 ResNet-50 모델
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # conv1
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x ~ conv5_x
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # 분류기
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        layers = [BottleneckBlock(in_channels, mid_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(mid_channels * 4, mid_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # conv1
        x = self.layer1(x)  # conv2_x
        x = self.layer2(x)  # conv3_x
        x = self.layer3(x)  # conv4_x
        x = self.layer4(x)  # conv5_x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 사용 예시
model = ResNet50(num_classes=1000)
dummy = torch.randn(1, 3, 224, 224)  # 이미지 1장
output = model(dummy)
print(output.shape)  # → torch.Size([1, 1000])
```

---

## 📊 데이터 흐름 추적 (Shape 변화)

ResNet-50 기준, 이미지 1장(`batch=1`)이 통과할 때의 텐서 shape 변화입니다.

```
입력 이미지
(1, 3, 224, 224)
        │
        ▼  [conv1: 7×7, stride 2, 64 filters]
(1, 64, 112, 112)      # 224÷2=112  |  채널: 3 → 64
        │
        ▼  [MaxPool: 3×3, stride 2]
(1, 64, 56, 56)        # 112÷2=56
        │
        ▼  [conv2_x: Bottleneck ×3, 채널 64→256]
(1, 256, 56, 56)       # 크기 유지  |  채널: 64×4=256
        │
        ▼  [conv3_x: Bottleneck ×4, stride 2, 채널 128→512]
(1, 512, 28, 28)       # 56÷2=28   |  채널: 128×4=512
        │
        ▼  [conv4_x: Bottleneck ×6, stride 2, 채널 256→1024]
(1, 1024, 14, 14)      # 28÷2=14   |  채널: 256×4=1024
        │
        ▼  [conv5_x: Bottleneck ×3, stride 2, 채널 512→2048]
(1, 2048, 7, 7)        # 14÷2=7    |  채널: 512×4=2048
        │
        ▼  [Global Average Pooling]
(1, 2048, 1, 1)  →  flatten  →  (1, 2048)
        │ 
        ▼  [FC Layer: 2048 → 1000]
(1, 1000)              # 1000개 클래스 점수
        │
        ▼  [Softmax]
(1, 1000)              # 합이 1인 확률값 분포
```

### 💡 핵심 패턴 2가지

| 패턴 | 설명 |
|------|------|
| **해상도 ↓ → 채널 ↑** | 이미지 크기가 절반이 될 때마다 채널 수는 두 배 → 계산량을 일정하게 유지하는 설계 원칙 |
| **병목 채널 흐름** | Bottleneck 블록 내부는 항상 **"압축 → 처리 → 복원"** 패턴 (예: 256 → 64 → 64 → 256) |

---

## 📚 참고 자료

- 📄 [원문 논문 (arXiv)](https://arxiv.org/abs/1512.03385)
- 🤗 [torchvision ResNet 구현](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- 📖 [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)

---

<div align="center">

**논문 읽고 직접 정리한 ResNet 요약입니다** 🧠

</div>
