# 🚀 AlexNet 논문 분석: ImageNet Classification with Deep CNNs
> **"딥러닝 혁명의 시작점"**
> AlexNet은 사람이 직접 규칙을 정하던 방식에서 벗어나, "깊은 신경망이 스스로 특징을 학습"하게 하여 컴퓨터 비전의 패러다임을 바꾼 모델입니다.

---

## 🧐 1. 배경: 기존 방식의 한계
* **수동 특징 추출 (Manual Feature Engineering):** 과거에는 사람이 직접 특징(SIFT, HOG 등)을 설계했으나, 복잡하고 다양한 현실 이미지를 모두 표현하기에는 한계가 있었습니다.
* **자원 및 데이터 부족:** 데이터셋의 규모가 작았고, 거대한 신경망을 학습시킬 만한 컴퓨팅 성능(GPU)이 대중화되지 않았던 시기였습니다.

---

## 💡 2. 혁신: AlexNet의 돌파구
AlexNet은 크게 세 가지 측면에서 기술적 돌파구를 마련했습니다.

### ① 구조 (Architecture)
* **깊이의 중요성:** 5개의 Convolutional 레이어와 3개의 Fully-connected 레이어로 구성하여, 신경망이 깊어질수록 더 복잡한 특징을 포착할 수 있음을 증명했습니다.

### ② 속도 (Speed)
* **ReLU & GPU:** `ReLU` 활성화 함수를 도입하고, 2개의 GPU를 병렬로 사용하는 기법을 통해 학습 속도를 비약적으로 향상시켰습니다.

### ③ 일반화 (Generalization)
* **과적합 해결:** `Dropout`과 `Data Augmentation(데이터 증강)` 기법을 적극적으로 도입하여, 깊은 모델의 고질적 문제인 과적합(Overfitting)을 효과적으로 방지했습니다.

---

## 🏆 3. 결과 및 결론
* **ILSVRC-2012 압도적 우승:** 2위(오류율 26.2%)와 격차를 크게 벌린 **오류율 15.3%**를 기록하며 전 세계를 놀라게 했습니다.
* **결론:** "데이터가 충분하고 신경망이 깊다면, 기계가 사람보다 더 정교하게 특징을 추출할 수 있다"는 것을 증명하며 **딥러닝 시대의 서막**을 열었습니다.

---

---

# 🧠 AlexNet: Deep Convolutional Neural Networks 구조 분석 및 구현

이 문서는 딥러닝 역사에서 상징적인 모델인 **AlexNet**의 핵심 구조를 분석하고, 이를 PyTorch 코드로 연결하여 정리한 가이드입니다.

---

## 🏗️ 1. 핵심 구조 설명 (Core Architecture)

### ① Convolutional Layer — 합성곱 레이어
* **비유:** 돋보기로 사진을 꼼꼼히 훑어보는 것. 작은 창문(필터)을 이미지 위에서 슬라이딩하며 패턴(엣지, 색깔, 질감)을 찾아냅니다.
* **입력값:** 이미지 데이터 `(batch, 채널 수, 높이, 너비)` 형태의 숫자 배열
* **출력값:** **특징 맵(Feature Map)** — 패턴이 어디에 있는지를 담은 숫자 배열
* **하이퍼파라미터:** 커널 크기(11×11, 5×5, 3×3), 필터 개수(96, 256, 384...), 스트라이드(stride), 패딩(padding)

### ② ReLU Nonlinearity — 렐루 활성화 함수 (핵심)
* **비유:** 신호등. 음수 값은 무조건 0(빨간불)으로 막고, 양수 값은 그대로 통과(초록불)시킵니다. 
* **공식:** $f(x) = \max(0, x)$
* **특징:** 기존에 쓰던 `tanh`나 `sigmoid`는 값이 커지면 기울기가 거의 0이 되어(기울기 소실) 학습이 느려집니다. ReLU는 이 문제가 없어서 **6배 빠른 학습**을 가능케 했습니다.
* **입력/출력:** 합성곱 연산 후 나온 숫자 → 음수는 0, 양수는 그대로 유지
* **하이퍼파라미터:** 없음 (수식이 고정되어 있음)

### ③ Max Pooling — 맥스 풀링
* **비유:** 사진을 축소할 때 각 구역에서 가장 눈에 띄는 픽셀만 남기는 것. 중요한 정보만 남기고 나머지는 버립니다.
* **AlexNet의 특징:** 일반적인 풀링과 달리 겹치는 영역을 포함하는 **Overlapping Pooling**을 사용했습니다. (창 크기 $z=3$, 이동 간격 $s=2$로 설정)
* **입력/출력:** ReLU를 통과한 특징 맵 → 크기가 줄어든 특징 맵
* **하이퍼파라미터:** 풀링 창 크기(z=3), 스트라이드(s=2)

### ④ Local Response Normalization (LRN) — 국소 반응 정규화 (핵심)
* **비유:** 학교에서 한 학생이 너무 튀면 주변 학생들이 상대적으로 눌리는 것. 이웃한 필터들 사이에서 경쟁을 시켜 너무 강한 반응을 억제합니다.
* **특징:** 실제 뇌 신경세포의 **"측면 억제(lateral inhibition)"** 현상을 모방한 기법입니다.
* **입력/출력:** ReLU를 통과한 특징 맵 → 정규화된 특징 맵 (같은 크기)
* **하이퍼파라미터:** $k=2, n=5, \alpha=10^{-4}, \beta=0.75$

### ⑤ Fully-Connected Layer — 완전연결 레이어
* **비유:** 앞에서 찾아낸 모든 특징들을 모아 최종 판단을 내리는 판사. *"귀가 뾰족하고 수염이 있으니 고양이!"*
* **입력/출력:** 앞 레이어에서 나온 벡터 (4096차원) → 다음 레이어 뉴런값 (4096차원 → 최종 1000차원)
* **하이퍼파라미터:** 뉴런 수 (4096, 4096, 1000)

### ⑥ Dropout — 드롭아웃 (핵심)
* **비유:** 팀 훈련에서 매번 랜덤하게 일부 선수를 쉬게 하는 것. 특정 선수에게만 의존하지 않고 모든 선수가 실력을 키우게 됩니다.
* **특징:** 학습 중에 뉴런을 **확률 50%로 무작위**로 꺼버립니다. 덕분에 모델이 특정 패턴에만 의존하는 **과적합(Overfitting)** 을 방지합니다.
* **입력/출력:** FC 레이어의 출력값 → 일부 뉴런이 0이 된 출력값
* **하이퍼파라미터:** 드롭아웃 확률 (p=0.5)

### ⑦ Softmax — 소프트맥스
* **비유:** 점수 환산기. "고양이일 확률 87%, 개일 확률 10%..."처럼 모든 클래스의 확률 합이 1이 되도록 변환합니다.
* **입력/출력:** 1000개의 점수 숫자 → 1000개 클래스 각각의 확률 (합계 = 1.0)
* **하이퍼파라미터:** 없음

---


## 🔗 2. 논문 ↔ 코드 연결 (PyTorch)

| 논문 내용 (Paper Details) | PyTorch Code Implementation |
| :--- | :--- |
| **"96 kernels of size 11×11×3, stride 4"** | `nn.Conv2d(3, 96, kernel_size=11, stride=4)` |
| **"ReLU nonlinearity"** | `nn.ReLU(inplace=True)` |
| **"Local Response Normalization"** | `nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)` |
| **"Overlapping max-pooling, z=3, s=2"** | `nn.MaxPool2d(kernel_size=3, stride=2)` |
| **"Dropout with probability 0.5"** | `nn.Dropout(p=0.5)` |
| **"three fully-connected layers, 4096 neurons"** | `nn.Linear(9216, 4096)`, `nn.Linear(4096, 4096)` |
| **"1000-way softmax"** | `nn.Linear(4096, 1000)` + `nn.Softmax(dim=1)` |

---
## 🚀 Live Demo on Hugging Face

이 프로젝트는 **Hugging Face Spaces**를 통해 실시간 데모를 제공합니다. AlexNet이 층(Layer)마다 추출하는 시각적 정보를 활용한 **Style Transfer**를 직접 체험해 보세요.

[👉 AlexNet Style Transfer 데모 바로가기](Goms2/AlexNet-based_Neural_Style_Transfer)

### ✨ 주요 기능
* **Content Extraction:** AlexNet의 Conv5 레이어에서 사물의 형태와 구조 정보를 추출합니다.
* **Style Extraction:** Conv1, Conv2 레이어에서 질감과 색상 패턴(Style)을 추출합니다.
* **Gram Matrix Optimization:** 각 레이어의 상관관계를 계산하여 스타일을 원본 이미지에 입힙니다.
