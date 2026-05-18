# 🤝 프로젝트 인계 — React + TS + Vite 통합 단계 (M4-B 부터)

> **이 문서를 받은 AI에게**: 이전 채팅방에서 진행하던 LLM 기반 음식 챗봇 프로젝트의 React 마이그레이션 작업입니다.
> 이 내용 다 읽고 **첫 응답에서 현재 상태 짧게 정리 → M4-B 작업 시작 신호** 보내주세요.
> 코드는 한 단계씩 — 한 번에 다 쏟아붓지 말 것.

---

## 👤 사용자 페르소나 (매우 중요)

- **한국인 비전공자**, AI 부트캠프 수강 중
- 파이썬 기초 약함 — **코드 읽기는 가능, 직접 작성은 어려움**
- AI 이론(CV/RNN/CNN) 기초 약함, 수학 수식 부담
- **목표**: 면접 대응 + 실무 일할 수 있게 + 포트폴리오
- 환경: Windows 11, Python 3.13, Node.js (최신 LTS), VS Code
- 프로젝트 경로: `C:\Users\User\Desktop\0505_LLM_Proect\food_chatbot\`
- 발표까지 **약 1주 남음**, 하루 6시간 가용 (~42시간)

### 응답 스타일 규칙 (반드시 지킬 것)

1. **중학생도 이해할 수 있게** 설명
2. **비유 → 개념 → 기술 설명** 순서
3. 어려운 용어는 반드시 풀어서 설명 (React, hook, TypeScript 같은 것도)
4. **정확성 > 빠짐없는 설명 > 이해하기 쉬움** — 정확성 최우선
5. **emote/action 표시 절대 금지** (예: `*손을 든다*` 같은 거)
6. 코드 한 번에 쏟아붓지 말기 — **한 단계씩**
7. 파일 수정 시 **통째로 새 파일** 줘서 갈아끼우게 (변경점은 별도 짚어주기)
8. 면접에서 자랑할 포인트 적극 짚어주기
9. SQLite 사용 (PostgreSQL/MySQL 아님)
10. 단계마다 끝에 `ask_user_input_v0` 으로 다음 진행 의사 확인

---

## 🍽️ 프로젝트 개요

**LLM 기반 개인 맞춤형 음식 챗봇** — K-Digital NVIDIA AI Academy 졸업 작품 (4주, 4인 팀)

- **팀명**: Be전공자들
- **현재 사용자**: 문승현 (LLM/RAG 담당이지만 역할 구분 없이 진행 중)
- **다른 팀원**: 장태영(기획/프론트), 엄선우(데이터), 심현우(백엔드)
- **계획서 핵심**: 사용자의 건강 상태(당뇨/고혈압 등)에 맞춰 음식 추천 + 의학 가이드라인 근거 인용 + **음식 점수 카드**

---

## ✅ 현재까지 완성된 것 (단계별)

### Phase 1: 백엔드 핵심 (완료)
| 영역 | 상태 |
|---|---|
| 음식 검색 (벡터+SQL, 29만 행) | ✅ |
| LLM Function Calling (5개 도구) | ✅ |
| 사용자 정보 관리 (CRUD + 자동 컨텍스트 주입) | ✅ |
| 대화 영구 저장 (세션 이어가기) | ✅ |
| 의학·영양 가이드라인 RAG (13 PDF · 8,294 청크) | ✅ |
| Function Call 시 RAG 강제 | ✅ |
| 자동 평가 시스템 (25 시나리오, 100% 통과) | ✅ |

### Phase 2: FastAPI 백엔드 (이전 채팅방에서 완료)
| 영역 | 상태 |
|---|---|
| FastAPI 골격 + 사용자별 챗봇 캐시 (lazy init) | ✅ |
| Lifespan 패턴 (서버 시작/종료 자원 관리) | ✅ |
| 스트리밍 응답 (SSE, sse-starlette) | ✅ |
| 처리 과정 노출 (tool_start/tool_end 이벤트) | ✅ |
| CORS 설정 (5173, 5500, 3000 등) | ✅ |
| `PATCH /users/{user_id}` 프로필 동기화 + 캐시 챗봇 프롬프트 갱신 | ✅ |
| **음식 점수 산출 도구 (하이브리드)** | ✅ |
| **food_scorer.py — 가이드라인 출처 추적까지** | ✅ |

### Phase 3: 단일 HTML 프론트 통합 (이전 채팅방에서 완료, 데모 백업으로 유지)
| 영역 | 상태 |
|---|---|
| 가짜 사용자(SAMPLE_USERS) → 백엔드 /users 동기화 | ✅ |
| 가짜 음식 엔진(FOOD_DB) 제거 → SSE 통합 | ✅ |
| 플랜 전환 시 update_user_profile 호출 | ✅ |
| 음식 점수 카드 (foods 배열) 렌더링 | ✅ |
| 디버그 로거 시스템 (`?debug=1`) | ✅ |

**위치**: `food_chatbot/static/index.html` — 데모 백업 + 백엔드 검증용으로 유지

### Phase 4: React + TS + Vite 마이그레이션 (진행 중)
| 단계 | 작업 | 상태 |
|---|---|---|
| M1 | Vite + React + TS 셋업 | ✅ |
| M2 | 폴더 구조 + 타입 정의 + API 클라이언트 | ✅ |
| M3 | useUsers 훅 + TopBar + Sidebar 골격 | ✅ |
| **M4-A** | **useProjects 훅 + 백엔드 동기화 + Sidebar 플랜 목록** | ✅ **방금 완료** |
| **M4-B** | **useChatStream 훅 + ChatPanel + Message + 입력창** | 🔜 **이제 시작** |
| M5 | 처리 과정 패널 (ProcessPanel) | 🔜 |
| M6 | Vercel + Railway 배포 | 🔜 |

---

## 🗂️ 코드베이스 인벤토리

### 백엔드 (`food_chatbot/`)

```
food_chatbot/
├── main.py                  ← FastAPI 진입점 (CORS + 모든 엔드포인트)
├── chatbot.py               ← FoodChatbot 클래스 (chat + chat_stream)
├── config.py                ← 전역 설정
├── food_search.py           ← FoodSearcher (음식 벡터 검색)
├── food_scorer.py           ← score_food_for_user (하이브리드, 가이드라인 출처)
├── guideline_search.py      ← GuidelineSearcher (가이드라인 RAG)
├── guideline_*.py           ← extractor/chunker/indexer/build
├── history_store.py         ← HistoryStore (사용자/세션/메시지 CRUD)
├── tools_schema.py          ← ALL_TOOLS (5개: search/guideline/get/update/evaluate)
├── prompts.py               ← SYSTEM_PROMPT_BASE
├── db_setup.py              ← user_data.db 초기화
├── evaluate_chatbot.py      ← 25 시나리오 평가
├── test_stream_api.py       ← SSE 백엔드 검증용 클라이언트
├── mydata.db                ← 음식 영양 222MB
├── user_data.db             ← 사용자/세션/메시지
├── chroma_db/               ← 벡터 DB (food_database + nutrition_guidelines)
├── nutrition_guidelines/    ← PDF/추출/청크 원본
├── static/
│   └── index.html           ← 단일 HTML (데모 백업)
└── web/                     ← ⭐ React 프로젝트 (M1~M4-A 진행됨)
    ├── package.json
    ├── tsconfig.json
    ├── vite.config.ts
    ├── .env                 ← VITE_API_BASE=http://localhost:8000
    ├── index.html
    ├── public/
    │   ├── logo_1.svg
    │   └── logo_2.svg
    └── src/
        ├── main.tsx
        ├── App.tsx          ← TopBar + Sidebar + 메인 placeholder
        ├── App.css          ← 글로벌 + TopBar + Sidebar 스타일
        ├── index.css        ← (비어있음)
        ├── components/
        │   ├── TopBar.tsx   ← 상단바 + 사용자 드롭다운
        │   └── Sidebar.tsx  ← 플랜 목록 + 대화 자리(placeholder)
        ├── hooks/
        │   ├── useUsers.ts       ← 백엔드 /users 동기화 + activeUser
        │   └── useProjects.ts    ← 사용자별 플랜 + PATCH 자동 동기화
        ├── lib/
        │   ├── api.ts            ← fetchUsers, updateUserProfile
        │   ├── utils.ts          ← assignColor, deriveInitials, makeId
        │   └── directions.ts     ← DIRECTIONS 5종 + getDirection
        └── types/
            ├── api.ts            ← BackendUser, SSEEvent (discriminated union), FoodScore
            └── chat.ts           ← User, Message, Conversation, Project, ProcRow
```

### 환경

- Python 3.13, venv 활성화됨
- 주요 패키지: `chromadb`, `sentence-transformers`, `anthropic`, `python-dotenv`, `pymupdf`, `fastapi`, `uvicorn`, `sse-starlette`
- `.env` 에 `ANTHROPIC_API_KEY`
- 모델: `claude-opus-4-7`, 임베딩: `jhgan/ko-sroberta-multitask`
- Node.js LTS 최신, npm
- React 18+, TypeScript, Vite 8

---

## 🔑 핵심 데이터 모델

### 백엔드 SSE 이벤트 (5종)
```typescript
type SSEEvent =
  | { event: 'text';        data: { chunk: string } }
  | { event: 'tool_start';  data: { name: string } }
  | { event: 'tool_end';    data: { name: string } }
  | { event: 'food_score';  data: FoodScore }
  | { event: 'done';        data: {} }
  | { event: 'error';       data: { message: string } };

type FoodScore = {
  name: string;
  score: number;            // 0-100
  level: 'good' | 'warn' | 'danger';
  detail: string;
  sources?: FoodScoreSource[];   // 가이드라인 출처 추적
};
```

### 프론트 도메인 모델
```typescript
type Message = {
  role: 'user' | 'bot';
  text: string;
  foods?: FoodScore[];      // 봇 메시지의 점수 카드들
  t: number;                // 타임스탬프
};

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
};

type Project = {
  id: string;
  name: string;
  profile: ProfileKey;      // 'diabetes' | 'hypertension' | ...
  condition: string;        // "당뇨" or "당뇨 + 고혈압"
  goal: string;
  conversations: Conversation[];
  activeConvId: string | null;
};
```

---

## 🚀 이번 단계: M4-B (챗 영역 SSE 통합)

### 목표
React 안에서 단일 HTML 의 sendMessage 로직을 재구현.

### 작업 분해 (이 순서로)

1. **대화(Conversation) 상태 관리** — `useProjects` 안에 대화 CRUD 추가
   - createConversation, deleteConversation, setActiveConv 같은 함수들
   - 활성 대화 ID 도 사용자별로 추적
   
2. **`useChatStream` 훅** — SSE 호출 + 파싱 + 이벤트 처리
   - `fetch(${API_BASE}/chat/stream)` POST
   - `ReadableStream + TextDecoder` 로 청크 수신
   - CRLF→LF 정규화 (Windows 호환)
   - `\n\n` 으로 메시지 분리, `event:` / `data:` 파싱
   - 콜백으로 이벤트 흘려보내기: `onText`, `onToolStart`, `onToolEnd`, `onFoodScore`, `onDone`, `onError`
   
3. **`<MessageInput />` 컴포넌트** — 입력창 + Enter/Shift+Enter + 한국어 IME 처리
   - `e.isComposing` 체크 필수 (한국어 조합 중 Enter 무시)
   
4. **`<FoodScoreCard />` 컴포넌트** — 점수 카드 (출처 칩 포함, 4-D 통합)
   - good/warn/danger 색상
   - sources 배열 있으면 작은 칩으로 출처 표시
   
5. **`<Message />` 컴포넌트** — 사용자/봇 메시지
   - 봇 메시지는 마크다운 (`marked` 라이브러리)
   - foods 배열 있으면 FoodScoreCard 들 렌더
   
6. **`<ChatPanel />` 컴포넌트** — 메시지 목록 + 입력창 통합
   - 자동 스크롤
   - 타이핑 인디케이터 (점 3개 애니메이션)
   
7. **`<Sidebar />` 의 대화 목록 채우기** — 현재 placeholder 자리
   - 활성 대화 강조, 클릭 시 전환, + 버튼으로 새 대화

8. **`<App />` 에 ChatPanel 연결**

### 라이브러리 추가
- `marked` (마크다운 → HTML) — `npm install marked`

### 알려진 함정
- **한국어 IME**: `e.isComposing` 또는 `e.keyCode === 229` 체크. 안 하면 조합 중 Enter 로 전송됨
- **CRLF 줄바꿈**: sse-starlette 가 Windows 에서 `\r\n` 보냄 → `.replace(/\r\n/g, '\n')` 정규화
- **flush 동작**: TextDecoder 는 `stream: true` 옵션 필수 (멀티바이트 한글 경계 처리)
- **DOM 재사용**: 봇 메시지는 매 토큰마다 새로 그리지 말고 텍스트만 갱신 (성능)

---

## 🎤 면접 무기 누적 정리 (현재까지)

### LLM/RAG 무기
- Function Calling 5종 (search/guideline/get/update/evaluate)
- Function Call 시 RAG 강제 (시스템 프롬프트 가이드)
- 25 시나리오 100% 통과, RAG 출처 인용률 100%
- LLM 자기교정 (검색 결과 품질 평가 후 재검색)

### 점수 산출 무기 (4단계 핵심)
- **하이브리드 평가**: LLM 정성 + 규칙 정량 보정
- 임계값: 학회 진료지침 일일 권고치 → 한 끼 환산
  - 탄수화물 50g (당뇨병학회 2023)
  - 당류 10g (WHO)
  - 나트륨 600mg (고혈압학회 2022)
  - 칼로리 500kcal (비만학회)
- 결정론적·재현 가능한 평가 (RAG 매번 호출하지 않은 이유)
- sources 배열로 출처 추적 가능 (traceability)

### 백엔드 무기
- 사용자별 챗봇 인스턴스 캐시 + lazy init (임베딩 모델 로딩 5초 → 사용자당 1회)
- Lifespan 패턴 (DB 연결 startup/shutdown 훅)
- 캐시 챗봇 프롬프트 핫 리로드 (인스턴스 재생성 X)
- PATCH + Pydantic `exclude_unset=True` (partial update)
- SSE 이벤트 분리 (text/tool/food_score) — 콘텐츠 vs 메타 채널
- HTTPException 으로 정확한 상태 코드 반환

### 프론트 무기 (단일 HTML)
- 스트리밍 중 DOM 재사용 (성능)
- 마크다운 점진적 렌더링
- CRLF/LF 호환 SSE 파서
- 한국어 IME isComposing 체크
- 로거 시스템 (`?debug=1` URL 토글)

### React 무기 (M1~M4-A)
- 모놀리식 HTML → 컴포넌트 분해 (관심사 분리)
- TypeScript discriminated union (SSE 이벤트 타입 안전)
- 커스텀 훅 (`useUsers`, `useProjects`)
- 함수형 setState (stale closure 방지)
- `useCallback` 메모이즈 (불필요한 리렌더링 방지)
- `useEffect` 클린업 (메모리 누수 방지)
- 백엔드↔프론트 모델 변환 어댑터
- 4가지 상태 처리 (loading/error/empty/success)
- localStorage 점진 마이그레이션 (`_v1` 버전 키)
- `as const` 리터럴 타입

---

## 🛠 작업 패턴 (이전 채팅방 흐름 — 유지하세요)

각 단계마다:
1. **비유로 개념 설명** (먼저)
2. **개념 정의** (어려운 용어 풀어서)
3. **코드 한 단계씩** — 통째 파일 줄 때는 변경점 별도 짚기
4. **핵심 포인트 4-5개** 짚기
5. **면접 단골 질문** 미리 풀이
6. **확인 포인트** 명시 (어떻게 됐을 때 정상인지)
7. **`ask_user_input_v0`** 으로 다음 진행 의사 확인

### 코드 전달 방식
- 짧은 코드 (~20줄): 본문에 코드 블록
- 긴 코드 (한 파일 전체): `create_file` 로 `/mnt/user-data/outputs/` 저장 후 `present_files` 로 제공
- 사용자가 "파일로 줘" 라고 명시적으로 요청하면 무조건 파일로

### 패치 적용 방식 (긴 기존 파일 수정 시)
- 너무 큰 파일은 `bash_tool` + `python` 스크립트로 정확한 문자열 매칭 + assert 검증 → 자동 패치
- 변경 후 grep 으로 확인
- 잘못 패치되면 사용자가 헤매니까 검증 단계 필수

---

## 🚦 현재 환경 확인 체크리스트

새 채팅방 시작 시:
- [ ] uvicorn 떠있나? (1번 터미널)
- [ ] Vite 떠있나? (2번 터미널, `food_chatbot/web/`)
- [ ] `http://localhost:5173/` 에서 M4-A 상태 보이나? (사이드바에 "기본" 플랜, 우측 사용자 메뉴)
- [ ] uvicorn 콘솔에 `♻️ 프로필 동기화` 로그 보이나? (M4-A 의 핵심 검증)

---

## 🎯 첫 응답에서 해야 할 것

이 문서를 받았다면 **첫 응답에서**:

1. **"이전 진행 상황 확인했어"** 로 시작
2. **현재 위치 짧은 표** — 완료된 단계와 다음 단계
3. **M4-B 짧은 비유** (예: "지금까지 챗봇의 뼈대를 세웠다면, M4-B 는 챗봇의 입과 귀를 다는 작업")
4. **M4-B 8개 하위 작업 미리보기** (위 작업 분해 참고)
5. **다음 진행 의사 확인** — `ask_user_input_v0`
   - A. 1번 (대화 상태 관리) 부터 시작
   - B. 환경 확인 먼저 (Vite/uvicorn 떠있는지)
   - C. M4-A 까지의 코드 다시 한 번 보여줘 (점검)

### 사용자가 이미 가진 파일 (요청하지 말 것)
- 백엔드 13개 코어 파일 (chatbot.py, main.py, tools_schema.py, food_scorer.py 등)
- React 프로젝트 `web/` 의 M4-A 완료된 모든 파일 (App.tsx, useUsers, useProjects, TopBar, Sidebar, directions.ts, api.ts, utils.ts, types/api.ts, types/chat.ts)

### 알려진 백엔드 이슈 (의식할 것)
- 임베딩 모델 로딩 5~7초 → 서버 시작 시 1회만, 사용자별 lazy init 이미 적용됨
- 다중 도구 호출 시 응답 30초+ 가능 → 스트리밍이 그래서 중요

---

준비됐으면 시작해줘. 첫 응답에서 M4-B 단계로 자연스럽게 진입.
