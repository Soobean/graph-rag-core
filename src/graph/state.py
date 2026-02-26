"""
Graph RAG 파이프라인 상태 정의

LangGraph의 상태 관리를 위한 TypedDict 정의
"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.auth.models import UserContext
from src.domain.types import GraphSchema, QueryPlan, ResolvedEntity, UnresolvedEntity

# Re-export constants
from src.graph.constants import (  # noqa: F401
    AGGREGATE_INTENTS,
    AVAILABLE_INTENTS,
    IntentType,
)


class GraphRAGState(TypedDict, total=False):
    """
    GraphRAG 시스템의 전체 상태
    """

    # ── 1. 입력 (Input) ────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str

    # ── 2. Query Understanding ─────────────────────────
    intent: IntentType
    intent_confidence: float
    entities: dict[str, list[str]]
    resolved_entities: list[ResolvedEntity]
    unresolved_entities: list[UnresolvedEntity]
    query_plan: QueryPlan

    # ── 3. Graph Retrieval ─────────────────────────────
    schema: GraphSchema
    cypher_query: str
    cypher_parameters: dict[str, Any]
    graph_results: list[dict[str, Any]]
    result_count: int

    # ── 4. Response ────────────────────────────────────
    response: str

    # ── 5. 메타데이터 및 에러 처리 ─────────────────────
    error: str | None
    execution_path: Annotated[list[str], operator.add]

    # ── 6. Vector Search / Cache ───────────────────────
    question_embedding: list[float] | None
    cache_hit: bool
    cache_score: float
    skip_generation: bool

    # ── 7. 접근 제어 (Access Control) ──────────────────
    user_context: UserContext | None
