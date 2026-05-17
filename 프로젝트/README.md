# 🍽️ LLM 음식 챗봇 — 데이터 백엔드

> Anthropic Claude API + Function Calling + RAG 기반의 개인 맞춤형 음식 추천 챗봇.

**팀**: Be전공자들 (4인)
**프로젝트**: K-Digital NVIDIA AI Academy 졸업 작품
**기간**: 4주

---

## 📋 프로젝트 개요

사용자의 건강 상태(당뇨·고혈압 등)와 목표(다이어트·근육증가 등)에 맞춰 **개인 맞춤 음식 추천**을 제공하는 챗봇. 의학·영양 가이드라인을 기반으로 **근거 있는 답변**을 생성한다.

### 핵심 기능

- 🍽️ 29만 개 음식 영양 정보 의미 검색 (벡터 + SQL 결합)
- 🤖 LLM 자연어 채팅 (Claude Opus 4.7)
- 👤 사용자 정보 자동 학습 (당뇨, 알레르기 등)
- 💾 대화 영구 저장 + 세션 이어가기
- 🏷️ 9개 영양 태그 자동 매핑 (저당, 저나트륨 등)
- 📚 **의학·영양 가이드라인 RAG** (13개 PDF · 8,294 청크) ⭐
- 🎯 **출처·페이지 자동 인용** (계획서 #4 충족) ⭐

---

## 🛠️ 기술 스택

| 영역 | 기술 |
|---|---|
| LLM | Anthropic Claude Opus 4.7 (Function Calling) |
| 임베딩 | jhgan/ko-sroberta-multitask (768차원) |
| Vector DB | ChromaDB 1.5+ (2개 컬렉션: 음식 + 가이드라인) |
| RDB | SQLite (음식 영양 + 사용자/대화) |
| PDF 처리 | **PyMuPDF (fitz)** ⭐ |
| 언어 | Python 3.13 |

---

## 📦 설치

### 1) 가상환경

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2) 패키지 설치

```bash
pip install chromadb sentence-transformers anthropic python-dotenv pymupdf
```

### 3) 데이터 준비

| 파일 | 위치 | 설명 |
|---|---|---|
| `mydata.db` | 루트 | 음식 영양 DB (222MB, 29만 행) |
| `chroma_db/` | 루트 | 벡터 DB (음식 + 가이드라인 컬렉션) |
| `nutrition_guidelines/pdfs/` ⭐ | 루트 | 의학·영양 가이드라인 PDF 13개 |
| `user_data.db` | 자동 생성 | 사용자/세션/메시지 |

### 4) `.env` 파일 생성

```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 5) DB 셋업 (최초 1회)

```bash
python db_setup.py            # user_data.db 생성
python build_guideline_db.py  # 가이드라인 RAG DB 빌드 (멱등) ⭐
```

---

## 🚀 실행

### 챗봇 (인터랙티브 모드)

```bash
python chatbot.py
```

→ 사용자/세션 선택 메뉴 → 자연어로 대화.

```
🧑 > 당뇨인데 떡국 먹어도 돼?
🤖 Bot: 떡국 1인분 탄수화물이 70g이라 권장량(50g)보다 많아요.
        드시려면 반 그릇만, 단백질 반찬 곁들이세요.
        (출처: 2023 당뇨병 진료지침 p.84-86)
        ※ 본 정보는 참고용이며 실제 의료 상담을 대체하지 않습니다.
```

### 평가 (25개 시나리오 자동, RAG 포함)

```bash
python evaluate_chatbot.py
```

→ 정확도/시간/비용 측정 + RAG 메트릭 (실행 ~5분, ~₩2,000).

### 가이드라인 DB 재빌드 (PDF 추가 시)

```bash
python build_guideline_db.py
# → 멱등성: 처리된 PDF는 자동 스킵, 새 것만 처리
```

---

## 📊 평가 결과

| 지표 | 결과 |
|---|---|
| 전체 시나리오 성공률 | **25/25 (100%)** |
| **RAG 시나리오 성공률** ⭐ | **10/10 (100%)** |
| **출처 인용률** ⭐ | **8/8 (100%)** |
| 의료 면책 부착률 | 8/10 (80%) |
| 평균 응답 시간 | 15.94초 |
| 질문당 평균 비용 | ₩80.7 |
| 검증된 도구 | 4개 (search_food / search_guideline / get_user / update_user) |

---

## 📁 폴더 구조

```
food_chatbot/
├── 📌 코어 코드 (기존)
│   ├── config.py              # 전역 설정
│   ├── food_search.py         # 음식 검색 (벡터+SQL)
│   ├── history_store.py       # 사용자/세션/메시지 DB
│   ├── tools_schema.py        # LLM 도구 정의 (4개)
│   ├── prompts.py             # 시스템 프롬프트 (RAG 통합)
│   └── chatbot.py             # 챗봇 본체
│
├── 📚 RAG 코드 ⭐ 신규
│   ├── guideline_extractor.py # PDF → 텍스트
│   ├── guideline_chunker.py   # 텍스트 → 청크
│   ├── guideline_indexer.py   # 청크 → 임베딩 → ChromaDB
│   ├── guideline_search.py    # 가이드라인 검색
│   └── build_guideline_db.py  # 13개 PDF 일괄 빌드
│
├── 🛠️ 셋업/평가
│   ├── db_setup.py            # user_data.db 생성
│   └── evaluate_chatbot.py    # 자동 평가 (25 시나리오)
│
├── 💾 데이터
│   ├── mydata.db              # 음식 영양 (222MB)
│   ├── chroma_db/             # 벡터 DB
│   │   ├── food_database/     # 음식 컬렉션 (291,108개)
│   │   └── nutrition_guidelines/  ⭐ 가이드라인 컬렉션 (8,294개)
│   ├── user_data.db           # 자동 생성됨
│   └── nutrition_guidelines/  ⭐ 가이드라인 작업 폴더
│       ├── pdfs/              # 원본 PDF 13개
│       ├── extracted/         # 추출 텍스트 (.txt)
│       └── chunks/            # 청크 (.json)
│
├── 🔧 환경
│   ├── venv/                  # 가상환경
│   ├── .env                   # API 키 (커밋 금지!)
│   └── .gitignore
│
└── 📜 문서
    ├── README.md              # 이 파일
    └── RAG_HANDOVER_3.md      # 팀 인계 문서 (더 자세히)
```

---

## 🆘 트러블슈팅

| 증상 | 해결 |
|---|---|
| `ANTHROPIC_API_KEY 없음` | `.env` 파일 위치/내용 확인 |
| `Collection not found: food_database` | `chroma_db.tar.gz` 압축 풀었는지 확인 |
| `Collection not found: nutrition_guidelines` ⭐ | `python build_guideline_db.py` 실행 |
| `no such table` | `python db_setup.py` 실행 |
| `ModuleNotFoundError: fitz` ⭐ | `pip install pymupdf` |
| 한글 깨짐 (Windows) | `set PYTHONIOENCODING=utf-8` |
| 느린 응답 (>20초) | 첫 로딩은 모델 다운로드 때문, 두 번째부터 빠름 |
| PDF 추출 시 멈춤 | pypdf 대신 PyMuPDF 사용 확인 |

---

## 📚 더 자세히

- **팀 인계 / 아키텍처 / 남은 작업**: [RAG_HANDOVER_3.md](./RAG_HANDOVER_3.md)
- **계획서 원본**: 별첨

---

## 👥 팀

| 이름 | 역할 |
|---|---|
| 장태영 | 기획 / 프론트엔드 |
| **문승현** | **LLM / RAG (이번 작업)** |
| 엄선우 | 데이터 |
| 심현우 | 백엔드 (FastAPI) |
