# 🎨 AlexNet Style Transfer Pro

딥러닝 기반의 이미지 스타일 전이(Neural Style Transfer) 웹 애플리케이션입니다.  
원본 사진에 원하는 화풍(그림 스타일)을 입혀서 새로운 예술 작품을 만들어낼 수 있습니다.

---

## 📸 데모 화면

> 원본 사진 + 화풍 사진 → 결과 이미지

예시: 도시 사진 + 반 고흐의 별이 빛나는 밤 → 반 고흐 스타일로 변환된 도시 사진

---

## 📁 프로젝트 구조

```
📦 프로젝트 폴더
 ├── app.py              # 메인 애플리케이션 코드
 └── requirements.txt    # 필요한 라이브러리 목록
```

---

## 🛠️ 기술 스택

| 역할 | 라이브러리 |
|---|---|
| 딥러닝 프레임워크 | PyTorch |
| 사전 학습 모델 | AlexNet (torchvision) |
| 이미지 처리 | Pillow (PIL) |
| 웹 UI | Gradio |

---

## 🚀 설치 및 실행 방법

### 1단계 - 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2단계 - 라이브러리 설치

```bash
pip install -r requirements.txt
pip install gradio  # 본인 환경에 맞는 버전으로 별도 설치
```

> ⚠️ **참고:** GPU(CUDA)가 있는 환경에서는 훨씬 빠르게 실행됩니다. GPU가 없어도 CPU로 동작하지만 느릴 수 있습니다.

### 3단계 - 실행

```bash
python app.py
```

실행 후 브라우저에서 `http://localhost:7860` 으로 접속하면 됩니다.

---

## 📖 코드 상세 설명

### `requirements.txt` - 필요한 라이브러리

```
torch             # PyTorch: 딥러닝 계산 핵심 라이브러리
torchvision       # 사전 학습된 AlexNet 모델 불러오기에 사용
pillow            # PIL: 이미지 파일 열기/저장/변환에 사용
```

> 💡 **초보자 팁:** `requirements.txt`는 "이 프로젝트를 실행하려면 이 라이브러리들이 필요해요"라고 알려주는 목록 파일입니다. `pip install -r requirements.txt` 명령어 하나로 모두 설치할 수 있습니다.

---

### `app.py` - 메인 코드

코드는 크게 **3개의 파트**로 나뉩니다.

---

#### 🔷 파트 1: 모델 및 유틸리티 정의

##### `AlexNetStyleModel` 클래스 - 딥러닝 모델

```python
class AlexNetStyleModel(nn.Module):
    def __init__(self):
        self.model = models.alexnet(weights='IMAGENET1K_V1').features
        self.style_layers = [0, 3]    # 스타일 추출 레이어 번호
        self.content_layers = [10]    # 내용 추출 레이어 번호
```

- **AlexNet이란?** 2012년에 등장한 유명한 딥러닝 모델로, ImageNet 데이터셋으로 사전 학습되어 있습니다. 이미 수백만 장의 이미지를 학습했기 때문에 이미지의 특징을 잘 잡아냅니다.
- **`weights='IMAGENET1K_V1'`**: 인터넷에서 미리 학습된 가중치(weights)를 자동으로 다운로드해서 사용합니다.
- **`.features`**: AlexNet의 특징 추출 부분만 사용합니다. (분류 부분은 사용 안 함)
- **`style_layers = [0, 3]`**: 모델의 0번, 3번 레이어에서 '화풍(스타일)' 정보를 추출합니다. 앞쪽 레이어일수록 색감, 질감 같은 저수준 특징을 담습니다.
- **`content_layers = [10]`**: 10번 레이어에서 '내용(content)' 정보를 추출합니다. 깊은 레이어일수록 물체의 형태, 구조 같은 고수준 특징을 담습니다.

```python
def forward(self, x):
    for i, layer in enumerate(self.model):
        x = layer(x)
        if i in self.style_layers: style_features.append(x)
        if i in self.content_layers: content_features.append(x)
    return style_features, content_features
```

- 이미지(텐서 `x`)를 레이어에 순서대로 통과시키면서, 지정된 레이어에서 특징(feature)을 꺼내 저장합니다.

---

##### `image_loader` 함수 - 이미지 전처리

```python
def image_loader(image):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),      # 이미지 크기를 512x512로 통일
        transforms.ToTensor(),              # PIL 이미지 → PyTorch 텐서로 변환
        transforms.Normalize(               # 정규화 (ImageNet 평균/표준편차 사용)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return loader(image).unsqueeze(0)       # 배치 차원 추가: (C,H,W) → (1,C,H,W)
```

- 이미지를 모델이 처리할 수 있는 형태로 변환합니다.
- **정규화 값(0.485, 0.456 등)**: AlexNet이 학습할 때 사용한 ImageNet 데이터셋의 평균/표준편차입니다. 같은 값으로 정규화해야 모델이 올바르게 동작합니다.
- **`.unsqueeze(0)`**: 모델은 이미지를 '묶음(배치)' 단위로 처리하기 때문에 차원을 하나 추가합니다.

---

##### `gram_matrix` 함수 - 스타일 측정 핵심

```python
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)          # (채널, 높이*너비) 형태로 재배열
    gram = torch.mm(tensor, tensor.t())     # 행렬 곱셈으로 채널 간 상관관계 계산
    return gram.div(c * h * w)              # 크기로 나눠서 정규화
```

- **그람 행렬(Gram Matrix)** 은 이미지의 '스타일'을 수치로 표현하는 방법입니다.
- 각 채널(특징 맵) 사이의 **상관관계**를 계산합니다. 예를 들어 "파란색 줄무늬가 있을 때 거친 질감도 함께 나타나는가?" 같은 정보를 담습니다.
- 내용(무엇이 있는지)은 무시하고, 질감·패턴·색감 등 스타일 정보만 포착합니다.

---

#### 🔷 파트 2: 스타일 전이 실행 함수

```python
def run_style_transfer(content_img, style_img, iterations, lr, style_weight):
```

이 함수가 실제 스타일 전이를 수행하는 핵심입니다.

##### 준비 단계

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- GPU(CUDA)가 있으면 GPU를, 없으면 CPU를 자동으로 사용합니다.

```python
target_tensor = content_tensor.clone().requires_grad_(True)
```
- **target(타겟)** 은 최종 결과가 될 이미지입니다. 원본 이미지를 복사해서 시작합니다.
- **`requires_grad_(True)`**: 이 텐서는 학습(최적화) 대상임을 PyTorch에 알립니다. 일반 이미지 값들이 학습을 통해 조금씩 바뀝니다.

```python
optimizer = optim.Adam([target_tensor], lr=lr)
```
- **Adam 옵티마이저**: target 이미지의 픽셀값을 어떻게 업데이트할지 결정하는 알고리즘입니다.
- 보통 딥러닝에서는 모델의 가중치를 학습시키지만, 여기서는 **이미지 자체의 픽셀값**을 학습시키는 것이 독특한 포인트입니다.

##### 스타일/내용 특징 미리 추출

```python
with torch.no_grad():
    style_fts, _ = model(style_tensor)
    style_grams = [gram_matrix(ft) for ft in style_fts]
    _, content_fts = model(content_tensor)
```
- `torch.no_grad()`: 기준이 되는 스타일/내용 특징은 변하지 않으므로, 계산 자원을 아끼기 위해 그라디언트 계산을 끕니다.

##### 반복 학습 루프

```python
for _ in range(int(iterations)):
    target_style_fts, target_content_fts = model(target_tensor)
    
    # 내용 손실: target과 원본의 내용이 얼마나 다른지
    c_loss = torch.mean((target_content_fts[0] - content_fts[0])**2)
    
    # 스타일 손실: target과 화풍 이미지의 그람 행렬이 얼마나 다른지
    s_loss = 0
    for i, target_ft in enumerate(target_style_fts):
        target_gram = gram_matrix(target_ft)
        s_loss += torch.mean((target_gram - style_grams[i])**2)
    
    # 최종 손실 = 내용 손실 + (스타일 손실 × 스타일 강도)
    total_loss = c_loss + (s_loss * style_weight)
    
    optimizer.zero_grad()   # 이전 그라디언트 초기화
    total_loss.backward()   # 그라디언트 계산 (역전파)
    optimizer.step()        # target 이미지 픽셀값 업데이트
```

- **손실(Loss)**: "현재 target 이미지가 목표에서 얼마나 떨어져 있는가"를 숫자로 나타냅니다. 이 값을 줄이는 방향으로 target 이미지가 조금씩 변합니다.
- **내용 손실**: target이 원본의 구조/형태를 유지하도록 합니다.
- **스타일 손실**: target이 화풍 이미지의 질감/색감을 따라가도록 합니다.
- **`style_weight`**: 두 손실의 균형을 조절합니다. 값이 클수록 스타일을 더 강하게 적용합니다.

##### 결과 이미지 변환

```python
out = target_tensor.cpu().clone().detach().squeeze(0)
# 정규화 역변환 (원래 픽셀값 범위로 복원)
out = out * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) \
        + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
out = out.clamp(0, 1)           # 픽셀값을 0~1 범위로 제한
return transforms.ToPILImage()(out)
```
- 학습 결과 텐서를 다시 PIL 이미지(일반 이미지 파일)로 변환해서 반환합니다.
- 입력 시 적용했던 정규화를 반대로 풀어줍니다.

---

#### 🔷 파트 3: Gradio 웹 UI

```python
with gr.Blocks() as demo:
```
- **Gradio**는 Python 코드만으로 웹 UI를 만들 수 있는 라이브러리입니다.

##### 입력 컨트롤

| 컴포넌트 | 역할 |
|---|---|
| `gr.Image(type="pil")` | 이미지 업로드 박스 (원본, 화풍 각 1개) |
| `iter_slider` | 학습 횟수 (10~100, 기본값 30) |
| `lr_slider` | 학습률 (0.001~0.1, 기본값 0.02) |
| `weight_slider` | 스타일 강도 (1천~1천만, 기본값 10만) |
| `gr.Button` | "변환 시작" 버튼 |

##### 버튼 클릭 이벤트 연결

```python
run_btn.click(
    fn=run_style_transfer,          # 실행할 함수
    inputs=[content_input, style_input, iter_slider, lr_slider, weight_slider],
    outputs=output_image            # 결과를 표시할 곳
)
```
- 버튼을 누르면 UI의 입력값들을 모아서 `run_style_transfer` 함수를 호출하고, 반환된 이미지를 결과 칸에 표시합니다.

```python
demo.launch(ssr_mode=False)
```
- 웹 서버를 시작합니다. `ssr_mode=False`는 일부 환경에서 호환성 문제를 방지하기 위한 설정입니다.

---

## ⚙️ 파라미터 가이드

| 파라미터 | 범위 | 기본값 | 설명 |
|---|---|---|---|
| **학습 횟수 (Iterations)** | 10 ~ 100 | 30 | 높을수록 스타일이 정교해지나 시간이 오래 걸림. 추천: 30~50 |
| **학습률 (Learning Rate)** | 0.001 ~ 0.1 | 0.02 | 이미지 변화 속도. 너무 높으면 노이즈 발생 |
| **스타일 강도 (Style Weight)** | 1e3 ~ 1e7 | 1e5 | 높을수록 화풍이 강하게 적용됨 |

---

## 💡 알고리즘 동작 원리 요약

```
[원본 사진] ──────────────→ AlexNet ──→ content_features (형태/구조 정보)
                                                    ↓
[target 이미지] ──→ AlexNet ──→ target_features → 내용 손실(c_loss) 계산
                                    ↓
                              target_gram ──→ 스타일 손실(s_loss) 계산
                                                    ↑
[화풍 사진] ──────────────→ AlexNet ──→ Gram Matrix (스타일 정보)

total_loss = c_loss + s_loss × style_weight
↓
옵티마이저가 target 이미지의 픽셀을 조금씩 수정
↓
반복 (iterations 횟수만큼)
↓
최종 결과 이미지 출력
```

---

## 📦 의존성

```
torch
torchvision
pillow
```

---

## 📄 참고 자료

- [Neural Style Transfer 원본 논문 (Gatys et al., 2015)](https://arxiv.org/abs/1508.06576)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
- [Gradio 공식 문서](https://www.gradio.app/docs)
