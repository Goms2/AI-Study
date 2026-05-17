import torch                              # PyTorch 메인 라이브러리 (텐서 연산용)
import torch.nn as nn                     # 신경망 레이어들을 모아둔 모듈 (Linear, Embedding 등)


class BertEmbeddings(nn.Module):
    """3가지 임베딩을 합산: Token + Segment + Position"""
    # BERT는 입력 토큰을 3개의 임베딩(단어 의미 + 문장 구분 + 위치 정보)으로 표현하고 더해서 사용

    def __init__(self, vocab_size=30000, hidden_size=768,
                 max_len=512, num_segments=2):
        # vocab_size=30000  : 단어 사전 크기 (BERT-base는 약 30,522)
        # hidden_size=768   : 임베딩/은닉 벡터의 차원 (BERT-base 기본값)
        # max_len=512       : 한 입력에서 처리 가능한 최대 토큰 길이
        # num_segments=2    : 문장 구분용 (문장 A=0, 문장 B=1) → NSP 때문에 필요
        super().__init__()                # nn.Module의 초기화 호출 (필수)

        # 토큰 임베딩: 단어 ID → 의미 벡터 (vocab_size × hidden_size 크기의 룩업 테이블)
        self.token_emb    = nn.Embedding(vocab_size, hidden_size)

        # 세그먼트 임베딩: 첫 번째 문장(0)인지 두 번째 문장(1)인지 표현
        self.segment_emb  = nn.Embedding(num_segments, hidden_size)

        # 포지션 임베딩: 토큰의 위치 정보 (0~511 위치마다 별도 벡터를 학습)
        # ※ 원조 Transformer는 sin/cos 고정값이지만, BERT는 학습 가능한 임베딩 사용
        self.position_emb = nn.Embedding(max_len, hidden_size)

        # Layer Normalization: 임베딩 합산 후 학습 안정화를 위해 정규화
        self.layer_norm   = nn.LayerNorm(hidden_size)

        # Dropout 0.1: 과적합 방지 (논문 기본값)
        self.dropout      = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        # input_ids:   (batch, seq_len) - 단어 ID 시퀀스
        # segment_ids: (batch, seq_len) - 0 또는 1로 채워진 문장 구분 시퀀스
        B, L = input_ids.shape            # B=배치 크기, L=시퀀스 길이

        # 위치 인덱스 [0, 1, 2, ..., L-1] 생성 후 배치 차원으로 확장
        # device 지정: 입력 텐서와 같은 디바이스(GPU/CPU)에 생성해야 에러 없음
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        # unsqueeze(0): (L,) → (1, L)
        # expand(B, L): (1, L) → (B, L) - 메모리 복사 없이 뷰만 확장

        # 3가지 임베딩을 단순 합산 → BERT의 핵심 아이디어
        emb = (self.token_emb(input_ids)        # (B, L, H) 단어 의미
             + self.segment_emb(segment_ids)    # (B, L, H) 문장 구분
             + self.position_emb(positions))    # (B, L, H) 위치 정보

        # LayerNorm 후 Dropout 적용하여 반환 (학습 안정성 + 정규화 효과)
        return self.dropout(self.layer_norm(emb))


class BertLayer(nn.Module):
    """Transformer Encoder 1개 블록: Self-Attention + FFN"""
    # BERT는 이 블록을 12번(BERT-base) 또는 24번(BERT-large) 쌓아 올림

    def __init__(self, hidden_size=768, num_heads=12, ffn_size=3072):
        # num_heads=12 : Multi-Head Attention의 헤드 개수
        #                각 헤드는 768/12 = 64차원씩 담당
        # ffn_size=3072: FFN 중간 확장 차원 (보통 hidden_size의 4배)
        super().__init__()

        # Multi-Head Self-Attention
        # batch_first=True: 입력 shape을 (B, L, H) 형태로 받음 (편리)
        # dropout=0.1: attention 가중치에 dropout 적용
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.1, batch_first=True
        )

        # Attention 출력에 대한 LayerNorm (Add & Norm의 Norm 부분)
        self.norm1 = nn.LayerNorm(hidden_size)

        # Feed-Forward Network (위치별 독립적으로 적용되는 2층 MLP)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),   # 768 → 3072 (4배 확장)
            nn.GELU(),                          # ⚠️ ReLU 아님! BERT는 GELU 사용
                                                # GELU는 더 부드러운 곡선으로 성능 우수
            nn.Linear(ffn_size, hidden_size),   # 3072 → 768 (원래 차원 복귀)
            nn.Dropout(0.1),                    # 정규화
        )

        # FFN 출력에 대한 LayerNorm
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        # x: (B, L, H) - 이전 층 출력 또는 임베딩 결과
        # attention_mask: (B, L) - 패딩 토큰 위치를 True로 표시

        # ━━━ 1단계: Self-Attention (양방향) ━━━
        # x를 query, key, value로 모두 사용 → Self-Attention
        # 양방향: GPT와 달리 미래 토큰을 가리지 않고 모든 토큰끼리 정보 교환
        # key_padding_mask: 패딩 위치(True)는 attention에서 무시
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        # 반환값 두 번째(_)는 attention weights인데 여기선 안 씀

        # Residual Connection + LayerNorm
        # 원본 x에 attention 결과를 더하고(잔차 연결) 정규화
        # ※ 이는 "Post-LN" 방식 (원조 Transformer/BERT 스타일)
        x = self.norm1(x + attn_out)

        # ━━━ 2단계: Feed-Forward Network ━━━
        ffn_out = self.ffn(x)                   # 위치별 비선형 변환

        # 다시 Residual + LayerNorm
        x = self.norm2(x + ffn_out)

        return x                                # (B, L, H) - 다음 층의 입력


class BERT(nn.Module):
    """BERT-Base: L=12, H=768, A=12"""
    # L = num_layers (Transformer 블록 개수)
    # H = hidden_size (벡터 차원)
    # A = num_heads (어텐션 헤드 개수)
    # 파라미터 수 약 110M (1.1억 개)

    def __init__(self, vocab_size=30000, hidden_size=768,
                 num_layers=12, num_heads=12, ffn_size=3072):
        super().__init__()

        # 입력 임베딩 모듈 (Token + Segment + Position)
        self.embeddings = BertEmbeddings(vocab_size, hidden_size)

        # Transformer Encoder를 num_layers개 만들어 ModuleList로 보관
        # ※ ModuleList: 일반 Python list와 달리 PyTorch가 파라미터를 자동 추적
        self.encoder = nn.ModuleList([
            BertLayer(hidden_size, num_heads, ffn_size)
            for _ in range(num_layers)
        ])

        # ━━━ Pre-training Heads (사전학습용 출력층) ━━━
        # MLM (Masked Language Modeling): [MASK] 토큰 위치의 원래 단어 맞히기
        # 각 토큰 벡터(H) → vocab 크기의 logits로 변환
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

        # NSP (Next Sentence Prediction): 두 문장이 이어지는지(0/1) 예측
        # [CLS] 토큰 벡터(H) → 2개 클래스 logits
        self.nsp_head = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, segment_ids, attention_mask=None):
        # ━━━ 1. 임베딩 ━━━
        # 단어 ID를 벡터로 변환 (Token + Segment + Position 합산)
        x = self.embeddings(input_ids, segment_ids)     # (B, L, H)

        # ━━━ 2. Transformer 블록 L번 통과 ━━━
        # 각 층마다 self-attention과 FFN을 거치며 표현이 정교해짐
        for layer in self.encoder:
            x = layer(x, attention_mask)                # (B, L, H) 유지

        # ━━━ 3. 출력 추출 ━━━
        # [CLS] 토큰(시퀀스 맨 앞): 문장 전체를 대표하는 벡터
        # → 분류 태스크(NSP, 감정분석 등)에 사용
        C = x[:, 0, :]                # (B, H)

        # 전체 토큰 벡터: 토큰 단위 태스크(MLM, NER, QA 등)에 사용
        T = x                         # (B, L, H)

        # MLM logits: 각 위치마다 어떤 단어인지 확률 분포
        mlm_logits = self.mlm_head(T) # (B, L, vocab_size)

        # NSP logits: 두 문장 연결 여부 (IsNext / NotNext)
        nsp_logits = self.nsp_head(C) # (B, 2)

        # 4가지 모두 반환 → 사전학습/파인튜닝 시 필요한 것 골라 사용
        return mlm_logits, nsp_logits, C, T


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# === 사용 예시 ===
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

model = BERT()                                      # 기본값으로 BERT-base 생성

# 임의의 단어 ID 텐서 생성: 0~29999 범위, 모양 (배치=2, 길이=128)
input_ids   = torch.randint(0, 30000, (2, 128))

# 세그먼트 ID: 전부 0 → 단일 문장 입력으로 처리한다는 의미
# (NSP를 쓸 때는 첫 문장 부분 0, 두 번째 문장 부분 1로 채움)
segment_ids = torch.zeros(2, 128, dtype=torch.long)

# 순전파 실행
mlm_logits, nsp_logits, C, T = model(input_ids, segment_ids)

# 각 출력의 shape 확인
print(mlm_logits.shape)   # torch.Size([2, 128, 30000]) - 위치마다 단어 분포
print(nsp_logits.shape)   # torch.Size([2, 2])          - 배치마다 IsNext/NotNext
print(C.shape)            # torch.Size([2, 768])        - [CLS] 벡터
