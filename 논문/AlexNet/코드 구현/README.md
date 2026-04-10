# 🎨 AlexNet Style Transfer

AlexNet의 특징 맵(Feature Map)을 활용하여 한 이미지의 화풍을 다른 이미지에 입히는 **딥러닝 스타일 전이** 웹 앱입니다.

---

## 📸 예시

| Content Image (원본) | Style Image (화풍) | Result (결과) |
|:---:|:---:|:---:|
| 풍경 사진 | 모네 그림 | 모네 화풍의 풍경 사진 |

---

## 🧠 작동 원리

이 프로젝트는 **Neural Style Transfer** 기법을 사용합니다.

```
Content Image (원본 사진)
        +                  ──▶  AlexNet Feature Extractor  ──▶  최적화(Adam)  ──▶  결과 이미지
Style Image (화풍 사진)
```

1. **Content Loss** — 원본 사진의 형태(사물 구조)를 유지합니다.
   - AlexNet의 `Conv5 (layer 10)` 특징 맵 사용
2. **Style Loss** — 화풍 이미지의 질감·색감을 전이합니다.
   - AlexNet의 `Conv1 (layer 0)`, `Conv2 (layer 3)` 특징 맵 사용
   - **Gram Matrix**로 스타일 패턴을 수치화
3. **최적화** — Adam 옵티마이저로 30번 반복하며 두 손실을 동시에 줄여 나갑니다.

---

## 🗂️ 프로젝트 구조

```
📁 project/
├── app.py            # 메인 코드 (모델 정의 + Gradio UI)
└── requirements.txt  # 필요한 라이브러리 목록
```

---

## ⚙️ 설치 및 실행 방법

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. 라이브러리 설치

```bash
pip install -r requirements.txt
```

> 💡 GPU가 있으면 자동으로 CUDA를 사용합니다. 없어도 CPU로 실행됩니다.

### 3. 앱 실행

```bash
python app.py
```

실행 후 브라우저에서 `http://localhost:7860` 으로 접속하세요.

---

## 🖥️ 사용 방법

1. **Content Image** — 형태를 유지할 원본 사진을 업로드합니다.
2. **Style Image** — 화풍을 가져올 그림/사진을 업로드합니다.
3. **Submit** 버튼을 클릭하면 약 30회 최적화 후 결과 이미지가 나타납니다.

---

## 📦 사용 기술

| 기술 | 용도 |
|---|---|
| `PyTorch` | 딥러닝 모델 구성 및 학습 |
| `TorchVision` | AlexNet 사전학습 모델 로드 |
| `Pillow (PIL)` | 이미지 입출력 처리 |
| `Gradio` | 웹 UI 인터페이스 |

---

## 📝 주요 코드 설명

### AlexNet 특징 추출기

```python
class AlexNetStyleModel(nn.Module):
    def __init__(self):
        self.model = models.alexnet(weights='IMAGENET1K_V1').features
        self.style_layers = [0, 3]    # 스타일 추출: Conv1, Conv2
        self.content_layers = [10]    # 내용 추출: Conv5
```

ImageNet으로 사전학습된 AlexNet에서 특징 추출 부분만 가져와 사용합니다.

### Gram Matrix

```python
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(c * h * w)
```

스타일(질감, 색감)을 수치로 표현하기 위한 행렬입니다. 채널 간의 상관관계를 계산합니다.

---

## 🔧 파라미터 조절 팁

`app.py` 내 다음 값을 바꿔서 결과를 조절할 수 있습니다.

| 파라미터 | 위치 | 효과 |
|---|---|---|
| `1e5` (스타일 가중치) | `total_loss` 계산 부분 | 값을 높이면 화풍이 더 강하게 적용됨 |
| `30` (반복 횟수) | 학습 루프 | 값을 높이면 결과가 더 정교해지나 느려짐 |
| `lr=0.02` (학습률) | `optim.Adam` | 값을 높이면 변화가 빠르나 불안정해질 수 있음 |

---

## 📄 라이선스

MIT License
