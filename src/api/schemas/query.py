"""
Query API Schemas

질의 API 요청/응답 스키마 정의
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.api.schemas.explainability import ExplainableResponse


class QueryRequest(BaseModel):
    """질의 요청"""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="사용자 질문",
        examples=["Find all nodes connected to Python"],
    )
    session_id: str | None = Field(
        default=None,
        description="세션 ID (선택)",
    )
    include_explanation: bool = Field(
        default=False,
        description="추론 과정 포함 여부",
    )
    include_graph: bool = Field(
        default=False,
        description="그래프 데이터 포함 여부",
    )
    graph_limit: int = Field(
        default=200,
        ge=1,
        le=500,
        description="그래프 최대 노드 수",
    )


class QueryMetadata(BaseModel):
    """질의 응답 메타데이터"""

    intent: str = Field(description="분류된 의도")
    intent_confidence: float = Field(description="의도 분류 신뢰도")
    entities: dict[str, list[str]] = Field(default={}, description="추출된 엔티티")
    resolved_entities: list[dict[str, Any]] = Field(
        default=[], description="해석된 엔티티"
    )
    cypher_query: str = Field(default="", description="생성된 Cypher 쿼리")
    cypher_parameters: dict[str, Any] = Field(default={}, description="Cypher 파라미터")
    result_count: int = Field(default=0, description="결과 수")
    execution_path: list[str] = Field(default=[], description="실행 경로")
    error: str | None = Field(default=None, description="에러 내용")


class QueryResponse(BaseModel):
    """질의 응답"""

    success: bool = Field(description="성공 여부")
    question: str = Field(description="원본 질문")
    response: str = Field(description="생성된 응답")
    metadata: QueryMetadata | None = Field(default=None, description="메타데이터")
    error: str | None = Field(default=None, description="에러 메시지")
    explanation: ExplainableResponse | None = Field(
        default=None,
        description="추론 과정 및 그래프 데이터",
    )


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(description="서비스 상태")
    version: str = Field(description="API 버전")
    neo4j_connected: bool = Field(description="Neo4j 연결 상태")
    neo4j_info: dict[str, Any] | None = Field(
        default=None, description="Neo4j 서버 정보"
    )


class SchemaResponse(BaseModel):
    """스키마 응답"""

    node_labels: list[str] = Field(default=[], description="노드 레이블 목록")
    relationship_types: list[str] = Field(default=[], description="관계 타입 목록")
    indexes: list[dict[str, Any]] = Field(default=[], description="인덱스 목록")
    constraints: list[dict[str, Any]] = Field(default=[], description="제약 조건 목록")


def _resolve_forward_refs() -> None:
    """Forward reference 해결"""
    from src.api.schemas.explainability import ExplainableResponse  # noqa: F401

    QueryResponse.model_rebuild()


_resolve_forward_refs()
