import torch
import torch.nn as nn

class SplitAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.dev0 = torch.device("cuda:0")
        self.dev1 = torch.device("cuda:1")

        # --- Layer 1 & 2: 독립 연산 구간 (GPU간 통신 없음) ---
        # 채널을 48개, 128개씩 나누어 각 GPU가 자기 할 일만 합니다.
        self.conv1_g0 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ).to(self.dev0)

        self.conv1_g1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ).to(self.dev1)

        self.conv2_g0 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ).to(self.dev0)

        self.conv2_g1 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ).to(self.dev1)

        # --- Layer 3: 첫 번째 Cross-GPU Communication 발생 ---
        # Conv3는 이전 레이어(Conv2)의 모든 특징 맵(128+128=256채널)을 참조해야 합니다.
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1).to(self.dev0) 
        self.relu3 = nn.ReLU(inplace=True).to(self.dev0)

        # --- Layer 4 & 5: 다시 독립 연산 구간 ---
        # Conv3의 결과(384채널)를 다시 192개씩 쪼개서 각 GPU로 보냅니다.
        self.conv4_g0 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True)).to(self.dev0)
        self.conv4_g1 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.ReLU(inplace=True)).to(self.dev1)
        
        self.conv5_g0 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2)).to(self.dev0)
        self.conv5_g1 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2)).to(self.dev1)

        # --- Classifier: 전체 통합 구간 (Cross-GPU) ---
        # FC 레이어는 모든 특징을 모아서 고도 인지를 수행합니다. (주로 메모리가 큰 GPU 1에서 수행)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ).to(self.dev1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 입력 데이터를 각 GPU로 복사 (병렬 처리 시작)
        x_g0 = x.to(self.dev0, non_blocking=True)
        x_g1 = x.to(self.dev1, non_blocking=True)

        # 2. Conv1 & Conv2: 각 GPU에서 독립적으로 계산
        x_g0 = self.conv2_g0(self.conv1_g0(x_g0))
        x_g1 = self.conv2_g1(self.conv1_g1(x_g1))

        # 3. [통신 1] Conv3를 위해 특징 맵 통합
        # GPU 1의 데이터를 GPU 0으로 보내서 256채널로 합칩니다.
        x_combined = torch.cat([x_g0, x_g1.to(self.dev0)], dim=1)
        x_combined = self.relu3(self.conv3(x_combined))

        # 4. Conv4 & Conv5를 위해 다시 쪼개기
        # 384채널을 192채널씩 나눠서 각 GPU로 분산
        x_g0, x_g1 = torch.split(x_combined, 192, dim=1)
        x_g1 = x_g1.to(self.dev1) # GPU 1로 전송

        x_g0 = self.conv5_g0(self.conv4_g0(x_g0))
        x_g1 = self.conv5_g1(self.conv4_g1(x_g1))

        # 5. [통신 2] Classifier (FC 레이어) 진입을 위해 통합
        # 모든 특징을 GPU 1로 모아서 최종 분류 수행
        x_final = torch.cat([x_g0.to(self.dev1), x_g1], dim=1)
        x_final = torch.flatten(x_final, 1)
        
        return self.classifier(x_final)
