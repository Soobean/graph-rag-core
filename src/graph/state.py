"""Graph RAG 파이프라인 상태 정의"""

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.auth.models import UserContext
from src.domain.types import GraphSchema, QueryPlan, ResolvedEntity, UnresolvedEntity
from src.graph.constants import (  # noqa: F401
    AGGREGATE_INTENTS,
    AVAILABLE_INTENTS,
    IntentType,
)


class GraphRAGState(TypedDict, total=False):
    # Input
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str

    # Query Understanding
    intent: IntentType
    intent_confidence: float
    entities: dict[str, list[str]]
    resolved_entities: list[ResolvedEntity]
    unresolved_entities: list[UnresolvedEntity]
    query_plan: QueryPlan

    # Graph Retrieval
    schema: GraphSchema
    cypher_query: str
    cypher_parameters: dict[str, Any]
    graph_results: list[dict[str, Any]]
    result_count: int

    # Response
    response: str

    # Metadata
    error: str | None
    execution_path: Annotated[list[str], operator.add]

    # Vector Search / Cache
    question_embedding: list[float] | None
    cache_hit: bool
    cache_score: float
    skip_generation: bool

    # Access Control
    user_context: UserContext | None
