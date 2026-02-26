"""
Domain Types

시스템 전반에서 사용되는 TypedDict 정의
dict[str, Any] 대신 명확한 타입을 사용하여 타입 안전성 확보
"""

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

# =============================================================================
# Graph Schema Types
# =============================================================================


class PropertySchema(TypedDict, total=False):
    """노드/관계 속성 스키마"""

    name: str
    type: str
    mandatory: bool
    sample_values: list[str]


class NodeSchema(TypedDict, total=False):
    """노드 레이블 스키마"""

    label: str
    properties: list[PropertySchema]
    count: int


class RelationshipSchema(TypedDict, total=False):
    """관계 타입 스키마"""

    type: str
    properties: list[PropertySchema]
    start_labels: list[str]
    end_labels: list[str]


class IndexSchema(TypedDict, total=False):
    """인덱스 스키마"""

    name: str
    type: str
    label: str
    properties: list[str]


class ConstraintSchema(TypedDict, total=False):
    """제약조건 스키마"""

    name: str
    type: str
    label: str
    properties: list[str]


class GraphSchema(TypedDict, total=False):
    """Neo4j 그래프 스키마 전체"""

    node_labels: list[str]
    relationship_types: list[str]
    nodes: list[NodeSchema]
    relationships: list[RelationshipSchema]
    indexes: list[IndexSchema]
    constraints: list[ConstraintSchema]


# =============================================================================
# Entity Types
# =============================================================================


class ExtractedEntity(TypedDict):
    """LLM이 추출한 엔티티"""

    type: str
    value: str
    normalized: str


class ResolvedEntity(TypedDict, total=False):
    """Neo4j에서 매칭된 엔티티"""

    id: str
    labels: list[str]
    name: str
    properties: dict[str, Any]
    match_score: float
    original_value: str


class UnresolvedEntity(TypedDict):
    """미해결 엔티티 (Neo4j에서 매칭 실패)"""

    term: str
    category: str
    question: str
    timestamp: str


# =============================================================================
# LLM Response Types
# =============================================================================


class IntentClassificationResult(TypedDict):
    """Intent 분류 결과"""

    intent: str
    confidence: float


class EntityExtractionResult(TypedDict):
    """엔티티 추출 결과"""

    entities: list[ExtractedEntity]


class IntentEntityExtractionResult(TypedDict):
    """통합 Intent 분류 + 엔티티 추출 결과"""

    intent: str
    confidence: float
    entities: list[ExtractedEntity]


class CypherGenerationResult(TypedDict, total=False):
    """Cypher 생성 결과"""

    cypher: str
    parameters: dict[str, Any]
    explanation: str


class PromptTemplate(TypedDict):
    """프롬프트 템플릿"""

    system: str
    user: str


# =============================================================================
# Neo4j Repository Result Types
# =============================================================================


class SubGraphNode(TypedDict):
    """서브그래프 노드 구조"""

    id: str
    labels: list[str]
    properties: dict[str, Any]


class SubGraphRelationship(TypedDict):
    """서브그래프 관계 구조"""

    id: str
    type: str
    start_node_id: str
    end_node_id: str
    properties: dict[str, Any]


class SubGraphResult(TypedDict):
    """서브그래프 조회 결과"""

    nodes: list[SubGraphNode]
    relationships: list[SubGraphRelationship]


# =============================================================================
# Pipeline Result Types
# =============================================================================


class FullState(TypedDict, total=False):
    """Explainability를 위한 상세 상태"""

    graph_results: list[dict[str, Any]]


class PipelineMetadata(TypedDict, total=False):
    """파이프라인 실행 메타데이터"""

    intent: str
    intent_confidence: float
    entities: dict[str, list[str]]
    resolved_entities: list[ResolvedEntity]
    cypher_query: str
    cypher_parameters: dict[str, Any]
    result_count: int
    execution_path: list[str]
    query_plan: dict[str, Any] | None
    error: str | None
    _full_state: FullState


class PipelineResult(TypedDict):
    """파이프라인 최종 실행 결과"""

    success: bool
    question: str
    response: str
    metadata: PipelineMetadata
    error: str | None


# =============================================================================
# Node Update Types
# =============================================================================


class IntentClassifierUpdate(TypedDict, total=False):
    intent: str
    intent_confidence: float
    execution_path: list[str]
    error: str | None


class EntityExtractorUpdate(TypedDict, total=False):
    entities: dict[str, list[str]]
    execution_path: list[str]
    error: str | None


class IntentEntityExtractorUpdate(TypedDict, total=False):
    intent: str
    intent_confidence: float
    entities: dict[str, list[str]]
    execution_path: list[str]
    error: str | None


class EntityResolverUpdate(TypedDict, total=False):
    resolved_entities: list[ResolvedEntity]
    unresolved_entities: list[UnresolvedEntity]
    entities: dict[str, list[str]]
    execution_path: list[str]
    error: str | None


class CypherGeneratorUpdate(TypedDict, total=False):
    schema: GraphSchema
    cypher_query: str
    cypher_parameters: dict[str, Any]
    execution_path: list[str]
    error: str | None


class GraphExecutorUpdate(TypedDict, total=False):
    graph_results: list[dict[str, Any]]
    result_count: int
    execution_path: list[str]
    error: str | None


class ResponseGeneratorUpdate(TypedDict, total=False):
    response: str
    messages: list["BaseMessage"]
    execution_path: list[str]
    error: str | None


class CacheCheckerUpdate(TypedDict, total=False):
    question_embedding: list[float] | None
    cache_hit: bool
    cache_score: float
    skip_generation: bool
    cypher_query: str
    cypher_parameters: dict[str, Any]
    execution_path: list[str]
    error: str | None


# Multi-hop Query Types


class QueryHop(TypedDict, total=False):
    step: int
    description: str
    node_label: str
    relationship: str
    direction: Literal["outgoing", "incoming", "both"]
    filter_condition: str | None


class QueryPlan(TypedDict, total=False):
    is_multi_hop: bool
    hop_count: int
    hops: list[QueryHop]
    final_return: str
    explanation: str


class QueryDecomposerUpdate(TypedDict, total=False):
    query_plan: QueryPlan
    execution_path: list[str]
    error: str | None


class QueryDecompositionResult(TypedDict, total=False):
    is_multi_hop: bool
    hop_count: int
    hops: list[QueryHop]
    final_return: str
    explanation: str


# =============================================================================
# Type Aliases
# =============================================================================

NodeUpdate = (
    IntentClassifierUpdate
    | EntityExtractorUpdate
    | IntentEntityExtractorUpdate
    | EntityResolverUpdate
    | CypherGeneratorUpdate
    | GraphExecutorUpdate
    | ResponseGeneratorUpdate
    | CacheCheckerUpdate
    | QueryDecomposerUpdate
)
