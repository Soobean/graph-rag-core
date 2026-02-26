"""
Explainability Service

AI 추론 과정 및 그래프 시각화 데이터 생성 로직을 담당하는 서비스
"""

from typing import Any

from src.api.schemas.explainability import (
    ExplainableGraphData,
    ThoughtProcessVisualization,
    ThoughtStep,
)
from src.api.schemas.visualization import GraphEdge, GraphNode
from src.api.utils.graph_utils import get_node_style, sanitize_props
from src.domain.types import FullState, PipelineMetadata

# ============================================
# 상수 정의
# ============================================

# 역할(role) → depth 기본 매핑
# 프론트엔드에서 depth 기반 레이아웃 계산 시 사용
ROLE_TO_DEFAULT_DEPTH: dict[str, int] = {
    "start": 0,  # 쿼리 엔티티 (검색 시작점)
    "end": 2,  # 결과 엔티티 (검색 결과)
}

# 노드 이름 → 단계 유형 매핑
NODE_TO_STEP_TYPE: dict[str, str] = {
    "intent_entity_extractor": "classification",
    "entity_resolver": "resolution",
    "cypher_generator": "generation",
    "graph_executor": "execution",
    "response_generator": "response",
    "clarification_handler": "response",
}

# 노드 이름 → 한국어 설명
NODE_DESCRIPTIONS: dict[str, str] = {
    "intent_entity_extractor": "질문 의도 분류 및 엔티티 추출",
    "entity_resolver": "엔티티 매칭 (Neo4j)",
    "cypher_generator": "Cypher 쿼리 생성",
    "graph_executor": "그래프 쿼리 실행",
    "response_generator": "응답 생성",
    "clarification_handler": "명확화 요청",
}


class ExplainabilityService:
    """Explainability 관련 로직을 처리하는 서비스"""

    def build_thought_process(
        self,
        metadata: PipelineMetadata,
        full_state: FullState | None,
    ) -> ThoughtProcessVisualization:
        """메타데이터에서 추론 과정 시각화 데이터 구축"""
        execution_path = metadata.get("execution_path", [])
        steps: list[ThoughtStep] = []

        for i, node_name in enumerate(execution_path):
            # 노드 이름 정리 (_error, _cached 접미사 제거)
            clean_name = node_name.replace("_error", "").replace("_cached", "")

            step = ThoughtStep(
                step_number=i + 1,
                node_name=node_name,
                step_type=NODE_TO_STEP_TYPE.get(clean_name, "execution"),
                description=NODE_DESCRIPTIONS.get(clean_name, node_name),
            )

            # 노드별 세부 정보 추가
            if clean_name == "intent_entity_extractor":
                intent = metadata.get("intent", "unknown")
                confidence = metadata.get("intent_confidence", 0.0)
                entities = metadata.get("entities", {})
                total = sum(len(v) for v in entities.values())
                step.output_summary = (
                    f"Intent: {intent} ({confidence:.0%}), 엔티티: {total}개"
                )
                step.details = {
                    "intent": intent,
                    "confidence": confidence,
                    "entities": entities,
                }

            elif clean_name == "entity_extractor":
                entities = metadata.get("entities", {})
                total = sum(len(v) for v in entities.values())
                step.output_summary = f"추출된 엔티티: {total}개"
                step.details = {"entities": entities}

            elif clean_name == "graph_executor":
                step.output_summary = f"결과: {metadata.get('result_count', 0)}건"

            elif clean_name == "cypher_generator":
                cypher = metadata.get("cypher_query", "")
                if cypher:
                    step.output_summary = f"Cypher 쿼리 생성 완료 ({len(cypher)}자)"

            steps.append(step)

        return ThoughtProcessVisualization(
            steps=steps,
            execution_path=execution_path,
        )

    def _assign_depth_values(self, nodes_map: dict[str, GraphNode]) -> None:
        """
        노드들에 depth 값 할당 (in-place 수정)

        x, y 좌표 계산은 프론트엔드에서 담당하며,
        백엔드는 depth 메타데이터만 제공합니다.

        Args:
            nodes_map: 노드 ID → GraphNode 맵 (depth가 할당됨)
        """
        for node in nodes_map.values():
            # depth가 설정되지 않은 경우 role 기반으로 기본값 할당
            if node.depth == 0 and node.role != "start":
                node.depth = ROLE_TO_DEFAULT_DEPTH.get(node.role or "end", 2)

    def build_graph_data(
        self,
        full_state: FullState,
        resolved_entities: list[dict[str, Any]],
        limit: int = 200,
    ) -> ExplainableGraphData:
        """graph_results에서 시각화 가능한 그래프 데이터 구축 (노드 + 엣지)"""
        graph_results = full_state.get("graph_results", [])

        nodes_map: dict[str, GraphNode] = {}
        edges_map: dict[str, GraphEdge] = {}

        query_entity_ids: list[str] = []
        result_entity_ids: list[str] = []

        # resolved_entities 이름 집합 (쿼리 시작점 판별용)
        resolved_names: set[str] = set()
        for entity in resolved_entities:
            name = entity.get("name", entity.get("original_value", ""))
            if name:
                resolved_names.add(name)

        def add_node(
            node_id: str,
            labels: list[str],
            name: str,
            props: dict[str, Any],
        ) -> None:
            """노드 추가 헬퍼 (depth는 role 기반, x/y는 프론트엔드에서 계산)"""
            if not node_id or node_id in nodes_map:
                return

            label = labels[0] if labels else "Node"

            # 역할 판별: resolved_entities에 매칭되면 시작점, 아니면 결과
            if name in resolved_names:
                role = "start"
                depth = 0
                query_entity_ids.append(node_id)
            else:
                role = "end"
                depth = 2
                result_entity_ids.append(node_id)

            nodes_map[node_id] = GraphNode(
                id=node_id,
                label=label,
                name=name or "Unknown",
                properties=sanitize_props(props),
                group=label,
                role=role,
                depth=depth,
                style=get_node_style(label),
            )

        def add_edge(
            edge_id: str,
            rel_type: str,
            source_id: str,
            target_id: str,
            props: dict[str, Any] | None = None,
        ) -> None:
            """엣지 추가 헬퍼"""
            if not edge_id or edge_id in edges_map:
                return
            if not source_id or not target_id:
                return

            edges_map[edge_id] = GraphEdge(
                id=edge_id,
                source=source_id,
                target=target_id,
                label=rel_type,
                properties=sanitize_props(props) if props else {},
            )

        # graph_results에서 노드/엣지 추출
        for row in graph_results[:limit]:
            for _key, value in row.items():
                if not isinstance(value, dict):
                    continue

                # 노드 형식 감지 (labels 속성이 있는 경우)
                if "labels" in value and isinstance(value.get("labels"), list):
                    fallback_id = f"node_{len(nodes_map)}"
                    elem_id = value.get("elementId", fallback_id)
                    node_id = str(value.get("id", elem_id))
                    labels = value.get("labels", [])
                    props = value.get("properties", {})
                    name = props.get("name", "") if isinstance(props, dict) else ""
                    node_props = props if isinstance(props, dict) else {}
                    add_node(node_id, labels, name, node_props)

                # 관계(Relationship) 형식 감지
                # Neo4j: {type: "HAS_SKILL", startNodeId/start, endNodeId/end}
                elif "type" in value and ("startNodeId" in value or "start" in value):
                    rel_type = value.get("type", "RELATED")
                    fallback_edge = f"edge_{len(edges_map)}"
                    edge_id = str(
                        value.get("id", value.get("elementId", fallback_edge))
                    )
                    source = str(value.get("startNodeId", value.get("start", "")))
                    target = str(value.get("endNodeId", value.get("end", "")))
                    props = value.get("properties", {})
                    add_edge(edge_id, rel_type, source, target, props)

            # 리스트 형식의 관계 처리 (path 쿼리 결과)
            for _key, value in row.items():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "type" in item:
                            rel_type = item.get("type", "RELATED")
                            fallback_edge = f"edge_{len(edges_map)}"
                            elem_id = item.get("elementId", fallback_edge)
                            edge_id = str(item.get("id", elem_id))
                            source = str(item.get("startNodeId", item.get("start", "")))
                            target = str(item.get("endNodeId", item.get("end", "")))
                            props = item.get("properties", {})
                            add_edge(edge_id, rel_type, source, target, props)

        for entity in resolved_entities:
            node_id = str(entity.get("id", ""))
            labels = entity.get("labels", [])
            name = entity.get("name", entity.get("original_value", ""))
            add_node(node_id, labels, name, entity.get("properties", {}))

        has_more = len(graph_results) > limit

        # depth 값 할당 (x, y 좌표는 프론트엔드에서 계산)
        self._assign_depth_values(nodes_map)

        # 유효한 엣지만 필터링 (source와 target이 모두 nodes_map에 존재하는 경우)
        valid_edges = [
            edge
            for edge in edges_map.values()
            if edge.source in nodes_map and edge.target in nodes_map
        ]

        return ExplainableGraphData(
            nodes=list(nodes_map.values())[:limit],
            edges=valid_edges,
            node_count=min(len(nodes_map), limit),
            edge_count=len(valid_edges),
            query_entity_ids=query_entity_ids,
            result_entity_ids=result_entity_ids,
            has_more=has_more,
        )
