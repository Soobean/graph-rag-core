"""
Graph RAG 파이프라인 상수 정의

Intent 분류 등 파이프라인 전반에서 참조되는 상수
Entity 타입은 Neo4j 스키마에서 자동 추출됩니다.
"""

from typing import Literal

# 메타 노드 필터링 기준 (스키마 추출 시 제외할 접두사)
META_NODE_PREFIXES: tuple[str, ...] = ("_", "__")

# 범용 Intent Type 정의
IntentType = Literal[
    "entity_search",  # 특정 엔티티 검색 (이름, 속성 기반)
    "relationship_search",  # 관계 탐색 (연결된 노드 찾기)
    "aggregation",  # 집계/통계 쿼리 (COUNT, AVG 등)
    "path_analysis",  # 경로 분석 (두 노드 간 연결)
    "comparison",  # 비교 분석 (순위, 차이 비교)
    "unknown",  # 분류 불가
]

# Intent 분류에 사용 가능한 의도 목록 (unknown 제외)
AVAILABLE_INTENTS: list[str] = [
    "entity_search",
    "relationship_search",
    "aggregation",
    "path_analysis",
    "comparison",
]

# 특정 엔티티 없이도 Cypher 생성이 가능한 집계/통계 intent
AGGREGATE_INTENTS: set[str] = {
    "aggregation",
    "comparison",
}
