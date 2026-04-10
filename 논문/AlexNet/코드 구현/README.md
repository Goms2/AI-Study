# 🎨 AlexNet Style Transfer

AlexNet의 특징 맵(Feature Map)을 활용하여 한 이미지의 화풍을 다른 이미지에 입히는 **딥러닝 스타일 전이** 웹 앱입니다.

---

## 📸 예시

| Content Image (원본) | Style Image (화풍) | Result (결과) |
|:---:|:---:|:---:|
| 풍경 사진 | 모네 그림 | 모네 화풍의 풍경 사진 |

---

## 🧠 작동 원리 (딥러닝 개념)

### Neural Style Transfer란?

**Neural Style Transfer(신경망 스타일 전이)** 는 두 장의 이미지를 입력받아,

- 하나의 이미지에서는 **형태(Content)** 를 가져오고
- 다른 이미지에서는 **화풍(Style)** 을 가져와

두 가지를 합친 새로운 이미지를 만드는 기술입니다.

```
[원본 사진]  ──▶ 형태 정보 추출 ──┐
                                   ├──▶ 합성 이미지 생성
[화풍 사진]  ──▶ 스타일 정보 추출 ─┘
```

---

### AlexNet을 특징 추출기로 사용하는 이유

**AlexNet** 은 이미지 분류를 위해 ImageNet 데이터셋으로 사전학습된 CNN(합성곱 신경망)입니다.  
사전학습 과정에서 AlexNet은 이미지의 다양한 시각적 특징을 자동으로 학습했습니다.

- **앞쪽 레이어 (Conv1, Conv2)** → 선, 색상, 질감 등 저수준 특징 학습 → **스타일 추출에 사용**
- **뒤쪽 레이어 (Conv5)** → 사물의 형태, 구조 등 고수준 특징 학습 → **내용 추출에 사용**

> 💡 AlexNet 자체를 학습시키는 게 아니라, 이미 학습된 AlexNet을 "측정 도구"로만 활용합니다.

---

### Content Loss (내용 손실)

원본 이미지와 결과 이미지가 **형태적으로 얼마나 다른지** 측정하는 값입니다.

```
Content Loss = (결과 이미지의 Conv5 특징값 - 원본 이미지의 Conv5 특징값)²
```

- 이 값이 작을수록 결과 이미지가 원본의 형태를 잘 유지하고 있다는 뜻입니다.

---

### Style Loss (스타일 손실) & Gram Matrix

화풍 이미지와 결과 이미지가 **질감·색감 면에서 얼마나 다른지** 측정하는 값입니다.

스타일은 "어떤 색이 있는가"보다 **"어떤 색들이 함께 나타나는가"** 의 패턴입니다.  
이를 수치화하기 위해 **Gram Matrix(그람 행렬)** 를 사용합니다.

```
Gram Matrix = 특징 채널들 간의 상관관계 행렬
```

예를 들어, 모네 그림에서 파란색과 흰색이 항상 함께 나타난다면,  
Gram Matrix에 그 패턴이 담기게 됩니다.

```
Style Loss = (결과 이미지의 Gram Matrix - 화풍 이미지의 Gram Matrix)²
```

---

### 최적화 과정 (어떻게 이미지가 만들어지는가?)

일반적인 딥러닝은 **모델의 가중치**를 학습합니다.  
이 프로젝트는 다릅니다. **결과 이미지 자체의 픽셀값**을 직접 최적화합니다.

```
1단계: 결과 이미지를 원본 사진으로 초기화
2단계: AlexNet으로 Content Loss + Style Loss 계산
3단계: Adam 옵티마이저가 Loss를 줄이는 방향으로 픽셀값 수정
4단계: 2~3단계를 30회 반복
```

즉, 반복할수록 결과 이미지는 원본의 형태를 유지하면서 화풍을 닮아갑니다.

---

## ⚙️ 설치 및 실행 방법

### 사전 준비

- **Python 3.8 이상** 이 설치되어 있어야 합니다.
- Python 설치 확인: 터미널에서 `python --version` 입력

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

> 💡 `git clone`은 GitHub 저장소를 내 컴퓨터로 복사하는 명령어입니다.

---

### 2. 가상환경 만들기 (권장)

라이브러리 충돌을 방지하기 위해 가상환경 사용을 권장합니다.

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Mac/Linux)
source venv/bin/activate
```

활성화되면 터미널에 `(venv)` 표시가 나타납니다.

---

### 3. 라이브러리 설치

```bash
pip install -r requirements.txt
pip install gradio
```

설치되는 라이브러리:

| 라이브러리 | 역할 |
|---|---|
| `torch` | 딥러닝 연산 엔진 |
| `torchvision` | AlexNet 사전학습 모델 제공 |
| `pillow` | 이미지 파일 읽기/저장 |
| `gradio` | 웹 UI 인터페이스 |

---

### 4. 앱 실행

```bash
python app.py
```

터미널에 아래 메시지가 나타나면 성공입니다.

```
Running on local URL: http://127.0.0.1:7860
```

브라우저에서 `http://localhost:7860` 으로 접속하세요.

---

### GPU 사용 (선택)

GPU(NVIDIA)가 있다면 CUDA 버전 PyTorch를 설치하면 훨씬 빠르게 실행됩니다.  
[PyTorch 공식 설치 페이지](https://pytorch.org/get-started/locally/)에서 본인 환경에 맞는 명령어를 확인하세요.

> 💡 GPU가 없어도 CPU로 정상 실행됩니다. 단, 속도가 느릴 수 있습니다.

---

## 🖥️ 사용 방법

1. **Content Image 업로드** — 형태를 유지할 원본 사진 (예: 내 사진, 풍경 사진)
2. **Style Image 업로드** — 화풍을 가져올 그림 (예: 모네, 반 고흐 작품)
3. **Submit** 클릭 → 약 30회 최적화 후 결과 이미지 출력

---

## 📝 주요 코드 설명

### 1. AlexNet 특징 추출기

```python
class AlexNetStyleModel(nn.Module):
    def __init__(self):
        super(AlexNetStyleModel, self).__init__()
        # ImageNet으로 사전학습된 AlexNet의 합성곱 레이어만 가져옵니다
        self.model = models.alexnet(weights='IMAGENET1K_V1').features
        
        self.style_layers = [0, 3]   # Conv1, Conv2: 저수준 특징 (스타일용)
        self.content_layers = [10]   # Conv5: 고수준 특징 (내용용)
```

`nn.Module`을 상속받아 커스텀 모델을 만듭니다.  
AlexNet 전체가 아닌 `.features` (합성곱 부분)만 가져와서 특징 추출기로 활용합니다.

```python
    def forward(self, x):
        style_features = []
        content_features = []
        
        for i, layer in enumerate(self.model):
            x = layer(x)                         # 레이어를 하나씩 통과
            if i in self.style_layers:
                style_features.append(x)          # 스타일 레이어면 저장
            if i in self.content_layers:
                content_features.append(x)        # 내용 레이어면 저장
                
        return style_features, content_features
```

이미지가 레이어를 하나씩 통과하면서, 지정된 레이어에서 중간 결과값을 수집합니다.

---

### 2. 이미지 전처리

```python
def image_loader(image):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),         # 모든 이미지를 512x512로 통일
        transforms.ToTensor(),                  # PIL 이미지 → PyTorch 텐서 변환
        transforms.Normalize(                   # AlexNet 학습 시 사용한 값으로 정규화
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return loader(image).unsqueeze(0)          # 배치 차원 추가 (1, C, H, W)
```

- **Resize**: 크기를 통일해야 연산이 가능합니다.
- **ToTensor**: 픽셀값(0~255)을 소수(0.0~1.0)로 변환합니다.
- **Normalize**: AlexNet이 학습된 환경과 동일하게 맞춰줘야 특징이 올바르게 추출됩니다.
- **unsqueeze(0)**: `(C, H, W)` → `(1, C, H, W)` 로 배치 차원을 추가합니다.

---

### 3. Gram Matrix

```python
def gram_matrix(tensor):
    _, c, h, w = tensor.size()       # c: 채널 수, h: 높이, w: 너비
    tensor = tensor.view(c, h * w)   # 2D로 펼치기: (채널, 픽셀 수)
    gram = torch.mm(tensor, tensor.t())  # 행렬 곱으로 채널 간 상관관계 계산
    return gram.div(c * h * w)       # 크기에 따른 스케일 보정
```

- `view(c, h * w)`: 3D 특징 맵을 2D 행렬로 펼칩니다.
- `torch.mm(tensor, tensor.t())`: 자기 자신과의 행렬 곱으로 채널 간 상관관계를 계산합니다.
- `div(c * h * w)`: 이미지 크기가 달라도 Loss 크기가 비슷하도록 정규화합니다.

---

### 4. 스타일 전이 메인 루프

```python
# 학습 대상: 모델이 아닌 결과 이미지 픽셀값
target_tensor = content_tensor.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target_tensor], lr=0.02)

for _ in range(30):
    target_style_fts, target_content_fts = model(target_tensor)
    
    # Content Loss
    c_loss = torch.mean((target_content_fts[0] - content_fts[0])**2)
    
    # Style Loss (각 스타일 레이어 합산)
    s_loss = 0
    for i, target_ft in enumerate(target_style_fts):
        target_gram = gram_matrix(target_ft)
        s_loss += torch.mean((target_gram - style_grams[i])**2)
    
    # 스타일 가중치 1e5: 스타일 손실의 규모가 작으므로 크게 보정
    total_loss = c_loss + (s_loss * 1e5)
    
    optimizer.zero_grad()   # 이전 기울기 초기화
    total_loss.backward()   # 역전파로 기울기 계산
    optimizer.step()        # 픽셀값 업데이트
```

- `requires_grad_(True)`: 이 텐서(결과 이미지)의 기울기를 계산하겠다는 선언입니다.
- `optimizer.zero_grad()` → `backward()` → `optimizer.step()` 은 PyTorch 학습의 기본 패턴입니다.

---

### 5. 역정규화 (결과 이미지 복원)

```python
out = target_tensor.cpu().clone().detach().squeeze(0)
# 정규화 역산: 원래 픽셀값으로 복원
out = out * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) \
        + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
out = out.clamp(0, 1)   # 픽셀값을 0~1 범위로 제한
```

전처리 시 정규화했던 값을 반대로 계산해 원래 이미지 범위로 되돌립니다.

---

## 🔧 파라미터 조절 팁

### 스타일 가중치 (`1e5`)

```python
total_loss = c_loss + (s_loss * 1e5)   # 여기를 수정
```

| 값 | 효과 |
|---|---|
| `1e4` (낮춤) | 원본 형태를 더 잘 유지, 화풍이 약하게 적용됨 |
| `1e5` (기본값) | 형태와 화풍의 균형 |
| `1e6` (높임) | 화풍이 매우 강하게 적용, 원본 형태가 흐릿해질 수 있음 |

> 💡 스타일 Loss의 절댓값이 Content Loss보다 훨씬 작기 때문에, 균형을 맞추기 위해 큰 가중치를 사용합니다.

---

### 반복 횟수 (`30`)

```python
for _ in range(30):   # 여기를 수정
```

| 값 | 효과 |
|---|---|
| `10~20` | 빠르지만 결과가 거칠 수 있음 |
| `30` (기본값) | 속도와 품질의 균형 |
| `100~300` | 품질이 더 좋아지지만 시간이 오래 걸림 |

> 💡 GPU 환경에서는 300회 이상도 부담 없이 실행할 수 있습니다.

---

### 학습률 (`lr=0.02`)

```python
optimizer = optim.Adam([target_tensor], lr=0.02)   # 여기를 수정
```

| 값 | 효과 |
|---|---|
| `0.005` (낮춤) | 변화가 천천히, 안정적이지만 수렴이 느림 |
| `0.02` (기본값) | 적당한 수렴 속도 |
| `0.1` (높임) | 빠르게 변하지만 결과가 불안정해질 수 있음 |

---

### 이미지 해상도 (`512`)

```python
transforms.Resize((512, 512))   # 여기를 수정
```

| 값 | 효과 |
|---|---|
| `256` | 빠른 처리, 낮은 해상도 |
| `512` (기본값) | 균형 잡힌 해상도 |
| `1024` | 고해상도, 메모리 많이 사용 (GPU 필요) |

---

## 📦 사용 기술

| 기술 | 용도 |
|---|---|
| `PyTorch` | 딥러닝 모델 구성 및 학습 |
| `TorchVision` | AlexNet 사전학습 모델 로드 |
| `Pillow (PIL)` | 이미지 입출력 처리 |
| `Gradio` | 웹 UI 인터페이스 |

---

## 🗂️ 프로젝트 구조

```
📁 project/
├── app.py            # 메인 코드 (모델 정의 + 스타일 전이 + Gradio UI)
└── requirements.txt  # 필요한 라이브러리 목록
```

---

## 📄 라이선스

MIT License
