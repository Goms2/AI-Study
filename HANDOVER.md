# 🤝 팀 인계 문서 — 음식 챗봇 백엔드

> **작성자**: 문승현 (LLM / RAG 담당)
> **목적**: 다음 세션 / 팀원이 이어받을 수 있도록 정리
> **완성도**: Phase 4 백엔드 코어 100% + RAG 통합 완료 (자동 평가 검증됨)

---

## 📊 현재 상태 한눈에

| 영역 | 상태 |
|---|---|
| 음식 검색 Function (벡터+SQL) | ✅ 완료, 9/9 테스트 통과 |
| LLM Function Calling | ✅ 완료, **4개 도구** |
| 사용자 정보 관리 | ✅ 완료 (CRUD + 자동 컨텍스트 주입) |
| 대화 히스토리 영구 저장 | ✅ 완료 (세션 이어가기 가능) |
| **의학·영양 가이드라인 RAG** ⭐ | ✅ **완료, 13개 PDF · 8,294 청크 인덱싱** |
| **Function Call 시 RAG 강제** (계획서 #4) ⭐ | ✅ **완료, 동시 호출률 100%** |
| 자동 평가 시스템 | ✅ 완료, **25/25 성공률** (RAG 10개 포함) |
| 시스템 프롬프트 튜닝 | ✅ 완료 (RAG + 출처 인용 + 의료 면책 강제) |

---

## 🗂️ 자산 인벤토리

### 1. 코드 파일 (13개 코어)

**기존 8개**

| 파일 | 라인~ | 역할 | 핵심 함수/클래스 |
|---|---|---|---|
| `config.py` | 130 | 전역 설정 | `CATEGORY_MAINS`, `NUTRITION_LABELS` |
| `food_search.py` | 280 | 음식 검색 (벡터+SQL) | `FoodSearcher.search()` |
| `history_store.py` | 290 | 사용자/세션/메시지 CRUD | `HistoryStore` (9개 메서드) |
| `tools_schema.py` | 200 | LLM 도구 JSON 스키마 | `ALL_TOOLS` (4개) |
| `prompts.py` | 100 | 시스템 프롬프트 (RAG 통합) | `SYSTEM_PROMPT_BASE` |
| `chatbot.py` | 300 | 챗봇 본체 (RAG 통합) | `FoodChatbot.chat()` |
| `db_setup.py` | 180 | user_data.db 생성 | `setup_database()` |
| `evaluate_chatbot.py` | 480 | 자동 평가 (25 시나리오) | `run_evaluation()` |

**RAG 신규 5개** ⭐

| 파일 | 라인~ | 역할 | 핵심 함수/클래스 |
|---|---|---|---|
| `guideline_extractor.py` | 100 | PDF → 텍스트 (PyMuPDF) | `extract_pdf_text()` |
| `guideline_chunker.py` | 200 | 텍스트 → 청크 (재귀적 분할) | `chunk_document()` |
| `guideline_indexer.py` | 200 | 청크 → 임베딩 → ChromaDB | `index_chunks()` |
| `guideline_search.py` | 200 | 가이드라인 검색 (ChromaDB only) | `GuidelineSearcher.search()` |
| `build_guideline_db.py` | 200 | 13개 PDF 일괄 빌드 (멱등성) | `main()` |

### 2. 데이터 자산

| 파일 | 크기 | 내용 |
|---|---|---|
| `mydata.db` | 222MB | 음식 영양 29만 행 |
| `chroma_db/food_database/` | ~1GB | 음식 컬렉션 (291,108 벡터) |
| `chroma_db/nutrition_guidelines/` ⭐ | ~50MB | **가이드라인 컬렉션 (8,294 청크)** |
| `user_data.db` | 자동 | users / sessions / messages |
| `nutrition_guidelines/pdfs/` ⭐ | ~100MB | 의학 PDF 13개 (원본) |
| `nutrition_guidelines/extracted/` ⭐ | ~10MB | 추출 텍스트 (디버깅·캐시) |
| `nutrition_guidelines/chunks/` ⭐ | ~20MB | 청크 JSON (캐시) |

### 3. 인덱싱된 가이드라인 PDF (13개) ⭐

| 분야 | PDF | 청크 |
|---|---|---|
| 고혈압 | 대한고혈압학회 2022, DASH | 381 |
| 당뇨병 | 대한당뇨병학회 2023, ADA, Diabetes Care 논문 2편 | 2,508 |
| 영양섭취기준 (KDRI) | 2025 KDRI 1·2·3권 (에너지/비타민/무기질) | 5,331 |
| 칼로리별 식단 | WhatsOnYourPlate 4종 (1200~2600cal) | 74 |
| **합계** | **13개 PDF** | **8,294 청크** |

---

## 🏗️ 아키텍처 (데이터 흐름)

```
사용자 입력
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│  FoodChatbot.chat()                                        │
│  1. user_data.db 에 user 메시지 저장                          │
│  2. Anthropic API 호출 (system + tools 4개 + messages)      │
│  3. LLM 도구 호출 (stop_reason='tool_use') →                │
│     ├ search_food_by_meaning                               │
│     │    → FoodSearcher (ChromaDB food_database + SQLite)  │
│     ├ search_nutrition_guidelines  ⭐ 신규                   │
│     │    → GuidelineSearcher (ChromaDB nutrition_guidelines)│
│     │    → min_similarity 0.45 필터로 노이즈 차단             │
│     ├ get_user_profile / update_user_profile                │
│     │    → HistoryStore                                     │
│  4. 결과를 LLM에 다시 → 최종 답변 (출처 인용 + 의료 면책)      │
│  5. user_data.db 에 assistant 메시지 저장                    │
└────────────────────────────────────────────────────────────┘
   │
   ▼
사용자에게 답변
```

---

## 🛠️ API 사용법

### 1. 챗봇 사용

```python
from chatbot import FoodChatbot

bot = FoodChatbot(user_id="u_xxx")
text = bot.chat("당뇨인데 떡국 먹어도 돼?")
# → 음식 영양(SQL+벡터) + 당뇨 가이드라인(RAG) 통합 답변, 출처 인용
```

### 2. 음식 검색만 (LLM 없이)

```python
from food_search import FoodSearcher
searcher = FoodSearcher()
results = searcher.search(query="단백질 많은 한식", min_nutrition={"protein_g": 20})
```

### 3. 가이드라인 검색만 (LLM 없이) ⭐ 신규

```python
from guideline_search import GuidelineSearcher

searcher = GuidelineSearcher()
results = searcher.search(
    query="당뇨 환자 운동 권장량",
    n_results=3,
    min_similarity=0.45,
)
# → [{chunk_id, source, pages, text, similarity}, ...]
```

### 4. 가이드라인 DB 빌드 (PDF 추가 시) ⭐ 신규

```bash
# 새 PDF를 nutrition_guidelines/pdfs/ 에 추가하고:
python build_guideline_db.py
# → 멱등성: 이미 처리된 PDF는 자동 스킵, 새 것만 처리
```

---

## 🤖 LLM 도구 (4개)

| 도구 | 트리거 | 입력 | 출력 |
|---|---|---|---|
| `search_food_by_meaning` | 음식 추천/검색 | query + 옵션 | 음식 리스트 |
| `search_nutrition_guidelines` ⭐ | 의학·영양 권고/근거 | query + n_results | 청크 (출처+페이지) |
| `get_user_profile` | 사용자 정보 조회 | (자동 주입) | 사용자 dict |
| `update_user_profile` | 사용자 정보 업데이트 | health_status/goals/notes | 업데이트 결과 |

> 💡 새 도구 추가: `tools_schema.py` + `chatbot._execute_tools()` 분기.

---

## 📊 평가 결과 (정량 데이터)

### 시나리오별 (25개, RAG 10개 포함)

| 카테고리 | 시나리오 | 성공 |
|---|---|---|
| 단순 검색 / 영양 태그 / min·max | 한식, 디저트, 저염, 당뇨, 살빼기 등 | 7/7 |
| 사용자 정보 | 당뇨, 알레르기 | 2/2 |
| 도구 호출 없음 | 인사, 감사 | 2/2 |
| 어려운 케이스 | 애매한 추천, 라면, 복합조건, 0건 | 4/4 |
| **RAG - 가이드라인 단독** ⭐ | 당뇨 운동, 비타민D | **2/2** |
| **RAG - 동시 호출 (계획서 #4)** ⭐ | 당뇨 음식, 떡국, 비타민D 음식 | **3/3** |
| **RAG - 의료 면책** ⭐ | 고혈압 식이요법 | **1/1** |
| **RAG - 0건 처리** ⭐ | 무관 쿼리, 음식 재시도 | **2/2** |
| **RAG - 사용자 컨텍스트** ⭐ | 1500kcal 식단 | **1/1** |
| **RAG - 출처 형식** ⭐ | 나트륨 권장량 | **1/1** |

### 총계

```
✅ 성공률: 25/25 (100%)

🎯 RAG 전용 메트릭:
   RAG 시나리오 성공률:           10/10 (100%)
   가이드라인 호출 시 출처 인용률:   8/8 (100%)  ← 계획서 #4 충족 정량 근거
   의료 면책 부착률:               8/10 (80%)

⏱️ 응답 시간:
   평균: 15.94초  (계획서 목표 15초 — 살짝 초과)
   최대: 33.78초  ([14] 복합 조건)
   최소: 3.11초   ([10] 인사)

💰 비용:
   총 입력 토큰:  313,327
   총 출력 토큰:   17,988
   총 비용:       ₩2,016.3 (25회)
   질문당 평균:   ₩80.7  (RAG 통합 전 ₩52.7 → +53%)
```

### 알려진 한계

- **다중 호출 케이스**: [13] 라면, [14] 복합 조건에서 도구 3~4번 호출. 시스템 프롬프트 "재호출 자제" 가 완전 보장 X. 비용/시간 최적화 여지.
- **응답 시간 살짝 초과**: 평균 15.94초 vs 목표 15초. 다중 호출 시나리오 영향.
- **영문 PDF 활용도**: WhatsOnYourPlate, ADA 등은 한국어 쿼리에서 유사도 낮음 (다국어 모델이지만 영어 매칭 약함).
- **입력 토큰 비중**: ~95% (시스템 프롬프트 + 도구 스키마 + 가이드라인 컨텍스트).

---

## 🚧 남은 작업 / 담당자

| # | 작업 | 담당 | 의존성 |
|---|---|---|---|
| 1 | **FastAPI 백엔드** (REST API 노출) | 심현우 | chatbot.py |
| 2 | **React 채팅 UI** (스트리밍, 처리과정 시각화) | 장태영 | FastAPI |
| 3 | ~~의학 가이드라인 RAG 통합~~ | ~~문승현~~ | ✅ **완료** |
| 4 | ~~Function Call 시 RAG 강제~~ | ~~문승현~~ | ✅ **완료** |
| 5 | 배포 (Vercel + Railway) | 팀 전체 | 위 모두 |

### 데이터/LLM 담당이 더 할 수 있는 작업 (옵션)

- 비용 최적화 (시스템 프롬프트 압축, 도구 description 다이어트)
- 다중 호출 강제 차단 (코드 단 호출 횟수 카운터)
- 추가 PDF 인덱싱 (지질이상증, 비만 진료지침 등)
- 평가 시나리오 확장 (현 25개 → 50개)

---

## 🎓 핵심 디자인 결정 (Why?)

### 1. **`category_main` / `category_sub` 사용 (vs `category1`/`category2`)**

mydata.db 의 식약처 원본 (64/1,390개) 은 너무 세분화 → 챗봇용 매핑 (8/61개) 사용.

### 2. **ChromaDB + SQLite 분리 (조인 패턴)**

ChromaDB(의미검색 후보 30) + SQLite(정확한 영양 필터). 각 DB 강점 활용. 임베딩엔 영양정보 미포함 (Single Source of Truth).

### 3. **`user_data.db` 별도 분리 (mydata.db 와)**

음식DB (정적, 222MB) ≠ 사용자DB (동적). 백업·마이그레이션 독립.

### 4. **`content` JSON + `content_text` 별도 저장**

LLM 복원 가능(JSON) + 사용자 표시 빠름(text).

### 5. **시스템 프롬프트에 사용자 정보 자동 주입**

매번 `get_user_profile` 호출 불필요 → 비용↓ 속도↑. `update_user_profile` 시 즉시 재빌드.

### 6. **`user_id` 자동 주입 (LLM에게 노출 안 함)**

도구 스키마에 user_id 인자 없음. LLM은 "누구의" 신경 안 써도 됨.

### 7. **prompts.py 분리**

시스템 프롬프트 자주 튜닝 → 본 코드 안 건드리고 prompts.py만 수정.

### 8. **PyMuPDF 사용 (vs pypdf)** ⭐

한국어 학회 PDF 처리 시 pypdf는 매우 느림 (`_handle_tm` 좌표 변환). PyMuPDF 96페이지 PDF → **0.1초** (pypdf는 미완료).

→ 라이센스 AGPL이지만 학습/포트폴리오용 문제 없음.

### 9. **재귀적 분할 (Recursive Splitting)** ⭐

우선순위: 빈 줄 → 줄바꿈 → 마침표 → 단어 → 글자. LangChain `RecursiveCharacterTextSplitter` 와 동일 패턴. 청크 500자, 오버랩 100자.

→ 자연스러운 문단 단위로 잘림.

### 10. **페이지 메타데이터 박아두기** ⭐

PDF 추출 시 `--- PAGE N ---` 마커 삽입. 청크 분할 후 `pages` 메타로 추출 (예: `"32-33"`).

→ "대한고혈압학회 진료지침 p.32" 같은 출처 인용 가능. 계획서 #4 충족 핵심.

### 11. **검색기 2개 분리 (FoodSearcher / GuidelineSearcher)** ⭐

같은 임베딩 모델 쓰지만 인스턴스 분리. 단일 책임 원칙 + 다른 컬렉션 + 다른 데이터 소스.

→ 메모리 부담 없음 (모델 weight 캐시 공유).

### 12. **min_similarity 임계값 0.45** ⭐

검색 결과 노이즈 차단. 무관 쿼리 → 0건 반환 → LLM이 의학 가이드라인을 억지로 인용 안 함.

→ "맛집 추천해줘" 같은 쿼리에서 정상 거절.

### 13. **멱등성 빌드 (캐싱 기반)** ⭐

`build_guideline_db.py` 는 .txt → .json → ChromaDB 단계별 캐시. 같은 명령 N번 안전.

→ PDF 1개 추가 시 그것만 처리. 중단 후 재실행 시 끝났던 부분부터.

---

## 🐛 알려진 이슈 / 주의사항

### 1. `food_database_test` 컬렉션 (leftover, 100건)
검색에 영향 X. 정리하려면 `client.delete_collection("food_database_test")`.

### 2. `disease_guideline` 빈 테이블 (mydata.db)
컬럼만 있고 데이터 없음. 팀 확인 필요.

### 3. `fiber_g` NULL 95.6%
식이섬유 데이터 거의 없음. `min_nutrition={"fiber_g": ...}` 검색은 사실상 작동 X.

### 4. `food_database` 일부 `category_sub` NULL (2.7%, 7,860건)
UI에서 "분류 미지정" 으로 처리 권장.

### 5. LLM 도구 재호출 (부분 개선) ⭐
RAG 통합 후 시스템 프롬프트에 "재호출 자제 + 0건 시 1회 재시도 OK" 명시. 다만 [13][14] 같은 복합 시나리오에서 3~4번 호출 발생.

해결 방법:
- (코드) `_call_llm` 에 도구 호출 횟수 카운터 + 강제 종료
- (프롬프트) "재호출 절대 금지" 더 강하게

### 6. 영문 PDF 활용도 낮음 ⭐
한국어 임베딩 모델은 다국어이긴 하지만 영어 PDF 매칭은 약함. 한국어 자료로 교체하거나, 영문 쿼리도 받도록 동시 임베딩 고려.

### 7. 응답 시간 살짝 초과 ⭐
평균 15.94초 (목표 15초). 다중 호출 시나리오 [13][14]가 33초까지 → 평균 끌어올림. 다중 호출 차단 시 해결.

---

## 💰 비용 추정 (월 단위, RAG 포함)

| 사용량 | 비용 |
|---|---|
| 1명 / 일 10회 사용 | 월 ~₩24,000 |
| 100명 / 일 10회 사용 | 월 ~₩2,400,000 |

> 💡 RAG 통합으로 비용 53% 증가 (₩52.7 → ₩80.7). 시스템 프롬프트 압축 + 가이드라인 컨텍스트 압축으로 30~40% 절감 가능.

---

## 📞 연락처

| 이름 | 역할 | 연락 |
|---|---|---|
| 장태영 | 기획 / 프론트엔드 | 010-9593-0897 |
| **문승현** | **LLM / RAG (이번 작업)** | **010-3662-4204** |
| 엄선우 | 데이터 | 010-6750-7494 |
| 심현우 | 백엔드 | 010-4362-7757 |

---

**🎯 다음 사람에게**: 이 백엔드는 단독으로 동작합니다.
- `python chatbot.py` — 인터랙티브 실행
- `python evaluate_chatbot.py` — 25 시나리오 자동 평가
- `python build_guideline_db.py` — PDF 추가 시 가이드라인 DB 재빌드 (멱등)

RAG 통합 완료 (계획서 #3, #4 충족). FastAPI 통합 시 `FoodChatbot` 클래스 그대로 사용 — 멀티스레드 대비 `check_same_thread=False` 적용됨. 가이드라인 검색기(`GuidelineSearcher`)도 동일 패턴이라 스레드 안전.

남은 작업: FastAPI 노출 → React UI → Vercel/Railway 배포. 🚀
