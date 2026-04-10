★ AlexNet 논문 분석 — ImageNet Classification with Deep CNNs

"딥러닝 혁명의 시작"</br>
AlexNet은 사람이 직접 규칙을 정하던 방식에서 벗어나, "깊은 신경망이 스스로 특징을 학습"하게 하여 컴퓨터 비전의 패러다임을 바꾼 모델

- 배경: 기존 방식의 한계</br>
   수동 특징 추출: 사람이 직접 특징(SIFT 등)을 설계했으나, 복잡한 현실 이미지를 담기엔 한계가 있음</br>
   자원 부족: 데이터셋 규모가 작았고, CNN을 돌릴만한 계산 성능이 부족했음</br>

- 혁신: AlexNet의 돌파구</br>
   구조: 5개의 Convolutional 레이어 + 3개의 Fully-connected 레이어로 구성 (깊이의 중요성 증명)</br>
   속도: ReLU 활성화 함수와 2개의 GPU 병렬 연산으로 학습 속도를 비약적으로 높임</br>
   성능: Dropout과 데이터 증강 기법을 도입해 과적합(Overfitting) 문제를 해결</br>

- 결과: 압도적 성과</br>
   ILSVRC-2012 우승: 2위(오류율 26.2%)와 격차를 크게 벌린 **15.3%**의 오류율 달성</br>

   결론: "신경망의 깊이가 성능을 결정한다"는 것을 증명하며 딥러닝 시대를 본격화함</br>

# 🧠 AlexNet: Deep Convolutional Neural Networks 구조 분석 및 구현

이 문서는 딥러닝 역사에서 상징적인 모델인 **AlexNet**의 핵심 구조를 분석하고, 이를 PyTorch 코드로 연결하여 정리한 가이드입니다.

---

## 🏗️ 3. 핵심 구조 설명 (Core Architecture)

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

### ③ Max Pooling — 맥스 풀링
* **비유:** 사진을 축소할 때 각 구역에서 가장 눈에 띄는 픽셀만 남기는 것. 중요한 정보만 남기고 나머지는 버립니다.
* **AlexNet의 특징:** 일반적인 풀링과 달리 겹치는 영역을 포함하는 **Overlapping Pooling**을 사용했습니다. (창 크기 $z=3$, 이동 간격 $s=2$로 설정)
* **입력/출력:** ReLU를 통과한 특징 맵 → 크기가 줄어든 특징 맵

### ④ Local Response Normalization (LRN) — 국소 반응 정규화 (핵심)
* **비유:** 학교에서 한 학생이 너무 튀면 주변 학생들이 상대적으로 눌리는 것. 이웃한 필터들 사이에서 경쟁을 시켜 너무 강한 반응을 억제합니다.
* **특징:** 실제 뇌 신경세포의 **"측면 억제(lateral inhibition)"** 현상을 모방한 기법입니다.
* **하이퍼파라미터:** $k=2, n=5, \alpha=10^{-4}, \beta=0.75$

### ⑤ Fully-Connected Layer — 완전연결 레이어
* **비유:** 앞에서 찾아낸 모든 특징들을 모아 최종 판단을 내리는 판사. *"귀가 뾰족하고 수염이 있으니 고양이!"*
* **입력/출력:** 앞 레이어에서 나온 벡터 (4096차원) → 다음 레이어 뉴런값 (최종 1000차원)

### ⑥ Dropout — 드롭아웃 (핵심)
* **비유:** 팀 훈련에서 매번 랜덤하게 일부 선수를 쉬게 하는 것. 특정 선수에게만 의존하지 않고 모든 선수가 실력을 키우게 됩니다.
* **특징:** 학습 중에 뉴런을 **확률 50%로 무작위**로 꺼버립니다. 덕분에 모델이 특정 패턴에만 의존하는 **과적합(Overfitting)**을 방지합니다.

### ⑦ Softmax — 소프트맥스
* **비유:** 점수 환산기. "고양이일 확률 87%, 개일 확률 10%..."처럼 모든 클래스의 확률 합이 1이 되도록 변환합니다.
* **입력/출력:** 1000개의 점수 숫자 → 1000개 클래스 각각의 확률 (합계 = 1.0)

---

## 🔗 4. 논문 ↔ 코드 연결 (PyTorch)

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

## 💻 5. 전체 모델 PyTorch 구현

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # ===== 합성곱 레이어 (Features) =====
        self.features = nn.Sequential(
            # Conv1: 224x224x3 -> 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 27x27x96
            
            # Conv2: 27x27x96 -> 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> 13x13x256
            
            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 -> 6x6x256 (After Pooling)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # ===== 완전연결 레이어 (Classifier) =====
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 9216)
        x = self.classifier(x)
        return x

# 모델 생성 및 테스트
model = AlexNet(num_classes=1000)
dummy_input = torch.randn(1, 3, 224, 224)
output = model(dummy_input)
print(f"Output shape: {output.shape}")  # torch.Size([1, 1000])
