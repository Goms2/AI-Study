import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import gradio as gr

# 1. AlexNet 기반 특징 추출기
class AlexNetStyleModel(nn.Module):
    def __init__(self):
        super(AlexNetStyleModel, self).__init__()
        # 사전 학습된 AlexNet 특징 추출부 로드
        self.model = models.alexnet(weights='IMAGENET1K_V1').features
        
        # 스타일 추출 레이어: Conv1(0), Conv2(3)
        # 내용 추출 레이어: Conv5(10)
        self.style_layers = [0, 3]
        self.content_layers = [10]
        
    def forward(self, x):
        style_features = []
        content_features = []
        
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.style_layers:
                style_features.append(x)
            if i in self.content_layers:
                content_features.append(x)
                
        return style_features, content_features

# 2. 이미지 처리 유틸리티
def image_loader(image):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return loader(image).unsqueeze(0)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram.div(c * h * w)

# 3. 스타일 전이 메인 함수
def run_style_transfer(content_img, style_img):
    if content_img is None or style_img is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNetStyleModel().to(device).eval()

    content_tensor = image_loader(content_img).to(device)
    style_tensor = image_loader(style_img).to(device)
    
    # 학습 대상 텐서 (원본 이미지 복사본)
    target_tensor = content_tensor.clone().requires_grad_(True).to(device)
    
    # 최적화 알고리즘 (Adam)
    optimizer = optim.Adam([target_tensor], lr=0.02)

    # 기준 특징 값 계산 (역전파 방지)
    with torch.no_grad():
        style_fts, _ = model(style_tensor)
        style_grams = [gram_matrix(ft) for ft in style_fts]
        _, content_fts = model(content_tensor)

    # 학습 루프 (안정성을 위해 30회 반복)
    for _ in range(30):
        target_style_fts, target_content_fts = model(target_tensor)
        
        # Content Loss: 사물의 형태 유지
        c_loss = torch.mean((target_content_fts[0] - content_fts[0])**2)
        
        # Style Loss: 질감과 화풍 유지
        s_loss = 0
        for i, target_ft in enumerate(target_style_fts):
            target_gram = gram_matrix(target_ft)
            s_loss += torch.mean((target_gram - style_grams[i])**2)
            
        total_loss = c_loss + (s_loss * 1e5) 
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    # 결과 이미지 복원 (역정규화)
    out = target_tensor.cpu().clone().detach().squeeze(0)
    out = out * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    out = out.clamp(0, 1)
    
    return transforms.ToPILImage()(out)

# 4. Gradio 인터페이스 구성
demo = gr.Interface(
    fn=run_style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image (원본 사진)"),
        gr.Image(type="pil", label="Style Image (화풍 사진)")
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="🎨 AlexNet Style Transfer",
    description="AlexNet의 특징 맵(Feature Map)을 활용하여 이미지의 스타일을 전이합니다."
)

if __name__ == "__main__":
    demo.launch()