import torch
import torch.nn as nn

# ① 병목 블록 (Bottleneck Block): 연산 효율을 높이기 위해 채널을 줄였다가 늘리는 구조
# ResNet-50, 101, 152와 같은 깊은 모델에서 사용됨
class BottleneckBlock(nn.Module):
    expansion = 4  # 최종 출력 채널이 중간 채널(mid_channels)의 4배가 됨

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()

        # [1x1 Conv] 입력 채널을 mid_channels로 압축 (연산량 감소 효과)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        # [3x3 Conv] 실제 공간적 특징을 추출 (stride에 따라 이미지 크기 조절)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        # [1x1 Conv] 압축했던 채널을 다시 expansion(4배)만큼 확장
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu  = nn.ReLU(inplace=True)

        # [Shortcut Connection] 입력 x를 출력에 더하기 위한 경로
        self.shortcut = nn.Sequential()
        
        # 만약 입력 채널과 출력 채널이 다르거나, stride가 1이 아니라면(크기가 변하면)
        # 1x1 Conv를 통해 입력 x의 형태를 출력과 동일하게 맞춰줌
        if stride != 1 or in_channels != mid_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * self.expansion)
            )

    def forward(self, x):
        identity = x  # 지름길(Shortcut)로 보낼 원본 입력값 저장

        # F(x) 연산: Residual Function
        out = self.relu(self.bn1(self.conv1(x)))   # 1x1 축소
        out = self.relu(self.bn2(self.conv2(out))) # 3x3 특징 추출
        out = self.bn3(self.conv3(out))            # 1x1 확장

        # 핵심: 잔차 학습 (F(x) + x)
        # 훈련 시 기울기가 identity 경로를 통해 원활하게 전달됨
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


# ② 전체 ResNet-50 모델 구조
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # [Stem Layer] 맨 처음 이미지의 크기를 빠르게 줄이는 구간 (224x224 -> 56x56)
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # [ResNet Stages] Bottleneck 블록들의 집합
        # layer1: 56x56 크기 유지, 64(x4) 채널, 3번 반복
        self.layer1 = self._make_layer(64,   64,  blocks=3, stride=1)
        # layer2: 28x28로 축소, 128(x4) 채널, 4번 반복
        self.layer2 = self._make_layer(256,  128, blocks=4, stride=2)
        # layer3: 14x14로 축소, 256(x4) 채널, 6번 반복
        self.layer3 = self._make_layer(512,  256, blocks=6, stride=2)
        # layer4: 7x7로 축소, 512(x4) 채널, 3번 반복
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # [Final Classifier] 최종 출력층
        # 전역 평균 풀링: 가로x세로를 평균내어 1x1로 만듦 (7x7x2048 -> 1x1x2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 1차원 벡터로 변환 후 최종 클래스 분류
        self.fc      = nn.Linear(2048, num_classes)

    # 블록들을 연결하여 하나의 스테이지(Layer)를 만드는 헬퍼 함수
    def _make_layer(self, in_channels, mid_channels, blocks, stride):
        layers = []
        # 스테이지의 첫 번째 블록: stride를 적용해 이미지 크기를 줄일 수 있음
        layers.append(BottleneckBlock(in_channels, mid_channels, stride=stride))
        
        # 두 번째 블록부터는 이미지 크기를 유지하며 쌓음 (입력 채널 = mid_channels * 4)
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(mid_channels * 4, mid_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 초기 컨볼루션 및 맥스풀링
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        # 2. 4개의 ResNet 스테이지 통과
        x = self.layer1(x) # conv2_x
        x = self.layer2(x) # conv3_x
        x = self.layer3(x) # conv4_x
        x = self.layer4(x) # conv5_x

        # 3. 풀링 및 분류기
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # (Batch, 2048, 1, 1) -> (Batch, 2048)
        x = self.fc(x)          # (Batch, num_classes)
        
        return x

# [테스트 코드]
if __name__ == "__main__":
    # 모델 생성 (예: CIFAR-10용이면 num_classes=10)
    model = ResNet50(num_classes=1000)
    
    # 가상의 이미지 데이터 (Batch=1, RGB=3, Height=224, Width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 모델 추론
    output = model(dummy_input)
    
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape}") # torch.Size([1, 1000])
