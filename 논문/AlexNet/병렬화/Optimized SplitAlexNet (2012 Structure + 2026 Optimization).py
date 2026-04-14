import torch
import torch.nn as nn

class OptimizedSplitAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # 장치 설정: GPU 0과 1에 분산 배치
        self.dev0 = torch.device("cuda:0")
        self.dev1 = torch.device("cuda:1")
        
        # [최적화] 데이터 전송 전용 스트림: 연산과 통신을 동시에 수행하기 위함
        self.comm_stream = torch.cuda.Stream(device=self.dev0)

        # --- Layer 1 & 2: 독립 병렬 연산 구간 ---
        # 논문 구조: 96개 채널을 48개씩 나누어 각 GPU에서 독립적으로 처리
        self.features_g0 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), # GPU 0 담당
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev0)

        self.features_g1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), # GPU 1 담당
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev1)

        # --- Layer 3: 첫 번째 Cross-GPU Communication (통합 구간) ---
        # 논문 구조: Conv3는 이전 레이어의 모든(128+128=256) 채널 정보를 참조함
        # 이를 위해 GPU 1의 데이터를 GPU 0으로 가져와서 여기서 연산
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1).to(self.dev0)
        self.relu3 = nn.ReLU(inplace=True).to(self.dev0)

        # --- Layer 4 & 5: 다시 독립 병렬 연산 구간 ---
        # 384개 채널을 다시 192개씩 쪼개어 각 GPU로 분산
        self.features2_g0 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev0)

        self.features2_g1 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(self.dev1)

        # --- Classifier: 고도 인지 및 최종 분류 (통합 구간) ---
        # FC 레이어는 메모리 소모가 크므로 상대적으로 여유 있는 GPU 1에서 수행
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096), # 128(g0) + 128(g1) = 256 채널
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        ).to(self.dev1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Step 1] 입력을 각 GPU로 복사 (병렬 처리 시작)
        # non_blocking=True를 사용하여 CPU가 기다리지 않고 다음 명령을 내리도록 최적화
        x_g0 = x.to(self.dev0, non_blocking=True)
        x_g1 = x.to(self.dev1, non_blocking=True)

        # [Step 2] Layer 1 & 2 연산 (각 GPU 독립 수행)
        x_g0 = self.features_g0(x_g0)
        x_g1 = self.features_g1(x_g1)

        # [Step 3] Conv3 통합을 위한 데이터 이동
        # GPU 1의 결과물을 GPU 0으로 전송 (이때 전용 스트림을 사용하여 병렬성 확보)
        with torch.cuda.stream(self.comm_stream):
            x_g1_to_dev0 = x_g1.to(self.dev0, non_blocking=True)
        
        # 현재 연산 스트림이 전송 완료를 기다리도록 설정 (동기화)
        torch.cuda.current_stream(self.dev0).wait_stream(self.comm_stream)
        
        # 두 데이터를 채널 방향(dim=1)으로 합쳐서 256채널 생성 후 Conv3 수행
        x_comb = torch.cat([x_g0, x_g1_to_dev0], dim=1)
        x_comb = self.relu3(self.conv3(x_comb))

        # [Step 4] Layer 4 & 5 수행을 위해 다시 데이터 분할
        # 384채널을 반으로 잘라 GPU 0과 1에 배분
        x_g0, x_g1 = torch.split(x_comb, 192, dim=1)
        x_g1 = x_g1.to(self.dev1, non_blocking=True) # 192채널을 GPU 1로 전송

        x_g0 = self.features2_g0(x_g0)
        x_g1 = self.features2_g1(x_g1)

        # [Step 5] 최종 Classifier 통합
        # 모든 특징 맵을 GPU 1로 모아 256채널 합치기
        x_final = torch.cat([x_g0.to(self.dev1, non_blocking=True), x_g1], dim=1)
        x_final = torch.flatten(x_final, 1) # FC 레이어 입력을 위해 평탄화
        
        # 최종 분류 결과 반환
        return self.classifier(x_final)
