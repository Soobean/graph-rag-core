"""
GraphEditService - 그래프 편집 비즈니스 로직

동적 스키마 기반 검증 + CRUD 오퍼레이션을 제공합니다.
도메인별 화이트리스트 대신 Neo4j 스키마에서 허용 라벨을 동적으로 조회합니다.
"""

import logging
from typing import Any

from src.domain.exceptions import EntityNotFoundError, GraphRAGError, ValidationError
from src.infrastructure.neo4j_client import Neo4jClient
from src.repositories.neo4j_repository import Neo4jRepository

logger = logging.getLogger(__name__)


# 시스템 메타데이터 보호 (사용자가 수정/삭제 불가)
PROTECTED_PROPERTIES = {"created_at", "created_by", "updated_at", "updated_by"}

# 검색 기본값
DEFAULT_SEARCH_LIMIT = 50

ANONYMOUS_ADMIN = "anonymous_admin"


class GraphEditConflictError(GraphRAGError):
    """그래프 편집 충돌 (중복 노드, 관계 존재 등)"""

    def __init__(self, message: str):
        super().__init__(message, code="GRAPH_EDIT_CONFLICT")


class GraphEditService:
    """
    그래프 편집 서비스

    Neo4jRepository를 통해 노드/엣지 CRUD를 수행하며,
    Neo4j 스키마에서 동적으로 허용 라벨/관계를 조회합니다.
    """

    def __init__(
        self,
        neo4j_repository: Neo4jRepository,
        neo4j_client: Neo4jClient | None = None,
    ):
        self._neo4j = neo4j_repository
        self._neo4j_client = neo4j_client
        self._allowed_labels: set[str] | None = None
        self._allowed_relationships: set[str] | None = None

    async def _get_allowed_labels(self) -> set[str]:
        """Neo4j 스키마에서 허용 라벨 동적 조회"""
        if self._allowed_labels is None:
            labels = await self._neo4j.get_node_labels()
            self._allowed_labels = set(labels)
        return self._allowed_labels

    async def _get_allowed_relationships(self) -> set[str]:
        """Neo4j 스키마에서 허용 관계 타입 동적 조회"""
        if self._allowed_relationships is None:
            rel_types = await self._neo4j.get_relationship_types()
            self._allowed_relationships = set(rel_types)
        return self._allowed_relationships

    def invalidate_schema_cache(self) -> None:
        """스키마 캐시 무효화 (노드/관계 추가 후 호출)"""
        self._allowed_labels = None
        self._allowed_relationships = None

    # ============================================
    # 노드 CRUD
    # ============================================

    async def create_node(
        self,
        label: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """노드 생성"""
        # 필수 속성 검증 (모든 라벨에 name 필수)
        if "name" not in properties or not str(properties["name"]).strip():
            raise ValidationError(
                "Property 'name' is required",
                field="name",
            )

        # 중복 방지
        name = str(properties["name"]).strip()
        properties["name"] = name
        is_duplicate = await self._neo4j.check_duplicate_node(label, name)
        if is_duplicate:
            raise GraphEditConflictError(
                f"Node with name '{name}' already exists in label '{label}'"
            )

        properties["created_by"] = ANONYMOUS_ADMIN

        result = await self._neo4j.create_node_generic(label, properties)
        if result is None:
            raise GraphEditConflictError(
                f"Node with name '{name}' already exists in label '{label}'"
            )

        # 새 라벨이 추가되었을 수 있으므로 캐시 무효화
        self.invalidate_schema_cache()

        logger.info(f"Node created: {label} '{name}' by {ANONYMOUS_ADMIN}")
        return result

    async def get_node(self, node_id: str) -> dict[str, Any]:
        """노드 조회 (ID 기반)"""
        node = await self._neo4j.find_entity_by_id(node_id)
        return {
            "id": node.id,
            "labels": node.labels,
            "properties": node.properties,
        }

    async def search_nodes(
        self,
        label: str | None = None,
        search: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> list[dict[str, Any]]:
        """노드 검색 (레이블, 이름 필터)"""
        return await self._neo4j.search_nodes(label=label, search=search, limit=limit)

    async def update_node(
        self,
        node_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """노드 속성 수정"""
        existing = await self._neo4j.find_entity_by_id(node_id)

        update_props, remove_keys = self._split_update_properties(properties)

        if "name" in update_props:
            await self._validate_name_update(update_props, existing)

        self._validate_remove_keys(remove_keys, existing)

        update_props["updated_by"] = ANONYMOUS_ADMIN

        result = await self._neo4j.update_node_properties(
            node_id, update_props, remove_keys or None
        )
        logger.info(f"Node updated: {node_id}")
        return result

    @staticmethod
    def _split_update_properties(
        properties: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """속성을 업데이트/삭제로 분리하고 시스템 메타데이터를 필터링"""
        update_props: dict[str, Any] = {}
        remove_keys: list[str] = []
        for key, value in properties.items():
            if key in PROTECTED_PROPERTIES:
                continue
            if value is None:
                remove_keys.append(key)
            else:
                update_props[key] = value
        return update_props, remove_keys

    async def _validate_name_update(
        self,
        update_props: dict[str, Any],
        existing: Any,
    ) -> None:
        """이름 변경 시 빈 값 및 중복 검증"""
        new_name = str(update_props["name"]).strip()
        if not new_name:
            raise ValidationError("name cannot be empty", field="name")
        update_props["name"] = new_name

        current_name = existing.properties.get("name", "")
        if new_name.lower() != str(current_name).lower():
            label = existing.labels[0] if existing.labels else ""
            if label:
                is_duplicate = await self._neo4j.check_duplicate_node(label, new_name)
                if is_duplicate:
                    raise GraphEditConflictError(
                        f"Node with name '{new_name}' already exists in label '{label}'"
                    )

    @staticmethod
    def _validate_remove_keys(remove_keys: list[str], existing: Any) -> None:
        """필수 속성 삭제 방지"""
        if not remove_keys:
            return
        if "name" in remove_keys:
            raise ValidationError(
                "Cannot remove required property: name",
                field="properties",
            )

    async def delete_node(
        self,
        node_id: str,
        force: bool = False,
    ) -> None:
        """노드 삭제"""
        result = await self._neo4j.delete_node_atomic(node_id, force=force)

        if result["not_found"]:
            raise EntityNotFoundError("Node", node_id)

        if not result["deleted"]:
            rel_count = result["rel_count"]
            raise GraphEditConflictError(
                f"Node has {rel_count} relationship(s). "
                f"Use force=true to delete with relationships."
            )

        logger.info(f"Node deleted: {node_id} (force={force})")

    # ============================================
    # 엣지 CRUD
    # ============================================

    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """엣지 생성"""
        # 소스/타겟 노드 존재 확인
        await self._neo4j.find_entity_by_id(source_id)
        await self._neo4j.find_entity_by_id(target_id)

        edge_props = dict(properties) if properties else {}
        edge_props["created_by"] = ANONYMOUS_ADMIN

        result = await self._neo4j.create_relationship_generic(
            source_id, target_id, relationship_type, edge_props
        )

        # 새 관계 타입이 추가되었을 수 있으므로 캐시 무효화
        self.invalidate_schema_cache()

        logger.info(
            f"Edge created: {relationship_type} "
            f"({source_id} -> {target_id}) by {ANONYMOUS_ADMIN}"
        )
        return result

    async def get_edge(self, edge_id: str) -> dict[str, Any]:
        """엣지 조회 (ID 기반)"""
        return await self._neo4j.find_relationship_by_id(edge_id)

    async def delete_edge(self, edge_id: str) -> None:
        """엣지 삭제"""
        deleted = await self._neo4j.delete_relationship_generic(edge_id)
        if not deleted:
            raise EntityNotFoundError("Edge", edge_id)
        logger.info(f"Edge deleted: {edge_id}")

    # ============================================
    # 스키마 정보
    # ============================================

    async def get_schema_info(self) -> dict[str, Any]:
        """편집 UI용 스키마 정보 반환 (동적 조회)"""
        labels = await self._get_allowed_labels()
        rel_types = await self._get_allowed_relationships()
        return {
            "allowed_labels": sorted(labels),
            "required_properties": {label: ["name"] for label in labels},
            "valid_relationships": sorted(rel_types),
        }
