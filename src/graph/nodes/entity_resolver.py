"""
Entity Resolver Node

추출된 엔티티를 Neo4j 그래프의 실제 노드와 매칭합니다.
"""

from datetime import UTC, datetime

from src.domain.types import EntityResolverUpdate, ResolvedEntity, UnresolvedEntity
from src.graph.nodes.base import DB_TIMEOUT, BaseNode
from src.graph.state import GraphRAGState
from src.repositories.neo4j_repository import Neo4jRepository


class EntityResolverNode(BaseNode[EntityResolverUpdate]):
    """엔티티 해석 노드"""

    def __init__(self, neo4j_repository: Neo4jRepository):
        super().__init__()
        self._neo4j = neo4j_repository

    @property
    def name(self) -> str:
        return "entity_resolver"

    @property
    def timeout_seconds(self) -> float:
        return DB_TIMEOUT

    @property
    def input_keys(self) -> list[str]:
        return ["entities"]

    async def _process(self, state: GraphRAGState) -> EntityResolverUpdate:
        """
        엔티티 해석 (그래프 노드 매칭)

        각 엔티티를 Neo4j에서 검색하여 매칭합니다.

        Args:
            state: 현재 파이프라인 상태

        Returns:
            업데이트할 상태 딕셔너리 (resolved + unresolved 엔티티 포함)
        """
        entities_map = state.get("entities", {})

        if not entities_map:
            self._logger.info("No entities to resolve")
            return EntityResolverUpdate(
                resolved_entities=[],
                unresolved_entities=[],
                execution_path=[f"{self.name}_skipped"],
            )

        entity_count = sum(len(values) for values in entities_map.values())
        self._logger.info(f"Resolving {entity_count} entities...")

        resolved: list[ResolvedEntity] = []
        unresolved: list[UnresolvedEntity] = []
        resolved_keys: set[tuple[str, str]] = set()  # (entity_type, value)
        question = state.get("question", "")
        now = datetime.now(UTC).isoformat()

        # 1단계: 엔티티 검색
        for entity_type, values in entities_map.items():
            for value in values:
                value = value.strip()
                if not value:
                    continue

                try:
                    matches = await self._neo4j.find_entities_by_name(
                        name=value,
                        labels=[entity_type]
                        if entity_type and entity_type != "Unknown"
                        else None,
                        limit=3,
                    )

                    if matches:
                        best_match = matches[0]
                        # 중복 방지
                        existing_ids = {r["id"] for r in resolved}
                        if best_match.id not in existing_ids:
                            resolved.append(
                                {
                                    "id": best_match.id,
                                    "labels": best_match.labels,
                                    "name": best_match.properties.get("name", value),
                                    "properties": best_match.properties,
                                    "match_score": 1.0,
                                    "original_value": value,
                                }
                            )
                        self._logger.debug(
                            f"Resolved '{value}' to node {best_match.id}"
                        )
                        resolved_keys.add((entity_type, value))

                except Exception as e:
                    self._logger.warning(f"Failed to resolve entity '{value}': {e}")

        # 2단계: unresolved 판단
        for entity_type, values in entities_map.items():
            for value in values:
                value = value.strip()
                if not value:
                    continue

                if (entity_type, value) not in resolved_keys:
                    unresolved.append(
                        UnresolvedEntity(
                            term=value,
                            category=entity_type,
                            question=question,
                            timestamp=now,
                        )
                    )
                    self._logger.info(
                        f"Unresolved entity: '{value}' (category: {entity_type})"
                    )

        self._logger.info(
            f"Resolved {len(resolved)} entities, {len(unresolved)} unresolved"
        )

        # 3단계: entities를 DB 실제 이름으로 보정
        # (예: "AI 연구소" → "AI연구소", Cypher generator가 정확한 이름 사용)
        corrected_entities: dict[str, list[str]] | None = None
        original_entities = state.get("entities", {})
        for r in resolved:
            db_name = r.get("name", "")
            original_value = r.get("original_value", "")
            if db_name and original_value and db_name != original_value:
                if corrected_entities is None:
                    corrected_entities = {
                        k: list(v) for k, v in original_entities.items()
                    }
                for entity_type, values in corrected_entities.items():
                    corrected_entities[entity_type] = [
                        db_name if v == original_value else v for v in values
                    ]
                self._logger.info(f"Corrected entity: '{original_value}' → '{db_name}'")

        result = EntityResolverUpdate(
            resolved_entities=resolved,
            unresolved_entities=unresolved,
            execution_path=[self.name],
        )

        if corrected_entities is not None:
            result["entities"] = corrected_entities

        return result
