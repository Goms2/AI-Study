import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # --- [특징 추출부: Feature Extraction] ---
        # Conv1: 입력 채널 3(RGB), 출력 채널 96, 커널 11x11, 스트라이드 4
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        
        # Conv2: 입력 96, 출력 256, 커널 5x5, 패딩 2 (크기 유지용)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        
        # Conv3: 입력 256, 출력 384, 커널 3x3, 패딩 1
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        
        # Conv4: 입력 384, 출력 384, 커널 3x3, 패딩 1
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        
        # Conv5: 입력 384, 출력 256, 커널 3x3, 패딩 1
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        
        # 공통 레이어: 가중치가 없어 재사용 가능
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2) # 국소 응답 정규화
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)                  # 겹치는 풀링(Overlapping)
        
        # --- [분류기: Classifier] ---
        self.dropout = nn.Dropout(p=0.5)                                   # 과적합 방지용 드롭아웃
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)                            # 첫 번째 전결합층 (9216 -> 4096)
        self.fc2 = nn.Linear(4096, 4096)                                   # 두 번째 전결합층 (4096 -> 4096)
        self.fc3 = nn.Linear(4096, num_classes)                            # 최종 출력층 (4096 -> 클래스 수)

    def forward(self, x):
        # 1층: Conv(11x11, s4) -> ReLU -> LRN -> MaxPool(3x3, s2)
        # Input: (3, 224, 224) -> Output: (96, 27, 27)
        x = self.pool(self.lrn(F.relu(self.conv1(x))))
        
        # 2층: Conv(5x5, p2) -> ReLU -> LRN -> MaxPool(3x3, s2)
        # Input: (96, 27, 27) -> Output: (256, 13, 13)
        x = self.pool(self.lrn(F.relu(self.conv2(x))))
        
        # 3층: Conv(3x3, p1) -> ReLU (풀링 없음)
        # Input: (256, 13, 13) -> Output: (384, 13, 13)
        x = F.relu(self.conv3(x))
        
        # 4층: Conv(3x3, p1) -> ReLU (풀링 없음)
        # Input: (384, 13, 13) -> Output: (384, 13, 13)
        x = F.relu(self.conv4(x))
        
        # 5층: Conv(3x3, p1) -> ReLU -> MaxPool(3x3, s2)
        # Input: (384, 13, 13) -> Output: (256, 6, 6)
        x = self.pool(F.relu(self.conv5(x)))
        
        # Flatten: 4차원 텐서를 2차원(배치, 특징)으로 펼침
        # (Batch, 256, 6, 6) -> (Batch, 9216)
        x = torch.flatten(x, 1)
        
        # 6층(FC1): Dropout -> Linear -> ReLU
        x = F.relu(self.fc1(self.dropout(x)))
        
        # 7층(FC2): Dropout -> Linear -> ReLU
        x = F.relu(self.fc2(self.dropout(x)))
        
        # 8층(FC3): Linear (최종 확률값 계산 전 로짓 상태)
        x = self.fc3(x)
        
        return x

# 모델 테스트
model = AlexNet(num_classes=1000)
print(model) # 모델의 전체 구조 출력
