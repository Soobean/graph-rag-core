# Graph RAG Core

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)](https://fastapi.tiangolo.com)
[![Neo4j](https://img.shields.io/badge/Neo4j-5-blue)](https://neo4j.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange)](https://github.com/langchain-ai/langgraph)

Neo4j 그래프 데이터베이스와 Azure OpenAI를 활용한 **도메인 독립적 Graph RAG 백엔드 템플릿**입니다.

자연어 질문을 받아 LangGraph 파이프라인을 통해 Cypher 쿼리로 변환하고, 그래프 DB 결과를 기반으로 자연어 응답을 생성합니다.

## 아키텍처

```
질문 → IntentEntityExtractor → EntityResolver → CypherGenerator → GraphExecutor → ResponseGenerator
                                     ↓ (unresolved)
                               ClarificationHandler
```

### 핵심 6노드 파이프라인
| 노드 | 역할 |
|------|------|
| IntentEntityExtractorNode | 의도 분류 + 엔티티 추출 (단일 LLM 호출) |
| EntityResolverNode | Neo4j에서 엔티티 매칭 |
| CypherGeneratorNode | Cypher 쿼리 생성 |
| GraphExecutorNode | Cypher 실행 |
| ResponseGeneratorNode | 자연어 응답 생성 |
| ClarificationHandlerNode | 명확화 질문 생성 |

## 빠른 시작

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에서 Neo4j + Azure OpenAI 설정 편집
```

### 2. Neo4j 실행

```bash
docker compose up -d
```

### 3. 의존성 설치

```bash
# uv 사용 (권장)
uv sync

# 또는 pip
pip install -e ".[dev]"
```

### 4. 서버 실행

```bash
uvicorn src.main:app --reload
```

API 문서: http://localhost:8000/docs

## 커맨드

```bash
# 서버 실행
uvicorn src.main:app --reload

# 테스트
pytest tests/ -v

# 린트
ruff check src/ tests/
ruff format --check src/ tests/

# 타입 체크
mypy src/
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | /api/v1/query | 자연어 질의 |
| POST | /api/v1/query/stream | 스트리밍 질의 |
| GET | /api/v1/health | 헬스체크 |
| GET | /api/v1/schema | 그래프 스키마 조회 |
| POST | /api/v1/graph/nodes | 노드 생성 |
| GET | /api/v1/graph/nodes | 노드 검색 |
| PATCH | /api/v1/graph/nodes/{id} | 노드 수정 |
| DELETE | /api/v1/graph/nodes/{id} | 노드 삭제 |
| POST | /api/v1/graph/edges | 엣지 생성 |
| DELETE | /api/v1/graph/edges/{id} | 엣지 삭제 |

## 프로젝트 구조

```
src/
├── main.py                 # FastAPI 엔트리포인트
├── config.py               # 설정 (환경변수)
├── dependencies.py         # DI 모듈
├── infrastructure/         # Neo4j 클라이언트
├── repositories/           # 데이터 접근 레이어
├── graph/                  # LangGraph 파이프라인
│   ├── pipeline.py         # 파이프라인 정의
│   ├── state.py            # 상태 타입
│   └── nodes/              # 파이프라인 노드들
├── domain/                 # 도메인 모델
├── services/               # 비즈니스 로직
├── api/                    # API 라우트 & 스키마
├── auth/                   # JWT 인증
├── prompts/                # LLM 프롬프트 템플릿 (YAML)
└── utils/                  # 유틸리티
```

## 커스터마이징

이 프로젝트를 특정 도메인에 맞게 커스터마이징하려면:

1. **프롬프트 수정** (`src/prompts/*.yaml`): 도메인에 맞는 인텐트 설명과 Cypher 예시 추가
2. **인텐트 확장** (`src/graph/constants.py`): 도메인 특화 인텐트 타입 추가
3. **노드 스타일** (`src/api/utils/graph_utils.py`): 시각화용 노드 라벨별 색상/아이콘 설정
4. **그래프 데이터 로드**: Neo4j에 도메인 데이터를 로드한 후 서버 시작

## 환경변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| NEO4J_URI | Neo4j 접속 URI | bolt://localhost:7687 |
| NEO4J_USER | Neo4j 사용자 | neo4j |
| NEO4J_PASSWORD | Neo4j 비밀번호 | password123 |
| AZURE_OPENAI_ENDPOINT | Azure OpenAI 엔드포인트 | - |
| AZURE_OPENAI_API_KEY | Azure OpenAI API 키 | - |
| AZURE_OPENAI_API_VERSION | API 버전 | 2024-10-21 |
| LIGHT_MODEL_DEPLOYMENT | 경량 모델 배포 이름 | gpt-4o-mini |
| HEAVY_MODEL_DEPLOYMENT | 고성능 모델 배포 이름 | gpt-4o |
| EMBEDDING_MODEL_DEPLOYMENT | 임베딩 모델 배포 이름 | text-embedding-3-small |
| AUTH_ENABLED | 인증 활성화 | false |
| JWT_SECRET_KEY | JWT 시크릿 키 | - |
| VECTOR_SEARCH_ENABLED | 벡터 검색 활성화 | true |
| VECTOR_SIMILARITY_THRESHOLD | 벡터 유사도 임계값 | 0.93 |
| GRAPH_EDIT_ENABLED | 그래프 편집 활성화 | true |

## 라이선스

MIT
