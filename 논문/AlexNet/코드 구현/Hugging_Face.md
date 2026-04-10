## 🚀 Live Demo on Hugging Face

이 프로젝트는 **Hugging Face Spaces**를 통해 실시간 데모를 제공합니다. AlexNet이 층(Layer)마다 추출하는 시각적 정보를 활용한 **Style Transfer**를 직접 체험해 보세요.

[👉 AlexNet Style Transfer 데모 바로가기](https://huggingface.co/spaces/Goms2/AlexNet-based_Neural_Style_Transfer)

### ✨ 주요 기능
* **Content Extraction:** AlexNet의 Conv5 레이어에서 사물의 형태와 구조 정보를 추출합니다.
* **Style Extraction:** Conv1, Conv2 레이어에서 질감과 색상 패턴(Style)을 추출합니다.
* **Gram Matrix Optimization:** 각 레이어의 상관관계를 계산하여 스타일을 원본 이미지에 입힙니다.
