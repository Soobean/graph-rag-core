"""Neo4j Repository — Facade over sub-repositories."""

from typing import Any

from src.domain.exceptions import QueryExecutionError
from src.infrastructure.neo4j_client import Neo4jClient
from src.repositories.neo4j_entity_repository import Neo4jEntityRepository
from src.repositories.neo4j_graph_crud_repository import Neo4jGraphCrudRepository
from src.repositories.neo4j_schema_repository import Neo4jSchemaRepository
from src.repositories.neo4j_types import NodeResult, RelationshipResult
from src.repositories.neo4j_vector_repository import Neo4jVectorRepository

__all__ = ["Neo4jRepository", "NodeResult", "RelationshipResult"]


class Neo4jRepository:
    """Neo4j Facade — 서브 레포지토리에 위임."""

    def __init__(self, client: Neo4jClient):
        self._client = client
        self._entity = Neo4jEntityRepository(client)
        self._schema = Neo4jSchemaRepository(client)
        self._vector = Neo4jVectorRepository(client)
        self._graph_crud = Neo4jGraphCrudRepository(client)

    # ── Entity ─────────────────────────────────────────────

    async def find_entities_by_name(self, name: str, labels: list[str] | None = None, limit: int = 10) -> list[NodeResult]:
        return await self._entity.find_entities_by_name(name, labels, limit)

    async def find_entity_by_id(self, entity_id: str) -> NodeResult:
        return await self._entity.find_entity_by_id(entity_id)

    async def get_neighbors(self, entity_id: str, relationship_types: list[str] | None = None,
                            direction: str = "both", depth: int = 1, limit: int = 50) -> list[dict[str, Any]]:
        return await self._entity.get_neighbors(entity_id, relationship_types, direction, depth, limit)

    async def get_relationships(self, entity_id: str, relationship_types: list[str] | None = None,
                                direction: str = "both", limit: int = 50) -> list[dict[str, Any]]:
        return await self._entity.get_relationships(entity_id, relationship_types, direction, limit)

    async def search_fulltext(self, search_term: str, index_name: str = "entityIndex", limit: int = 10) -> list[NodeResult]:
        return await self._entity.search_fulltext(search_term, index_name, limit)

    async def get_subgraph(self, entity_ids: list[str], max_depth: int = 1, limit: int = 100) -> Any:
        return await self._entity.get_subgraph(entity_ids, max_depth, limit)

    async def find_similar_nodes(self, embedding: list[float], index_name: str,
                                 exclude_ids: list[str] | None = None, limit: int = 5, threshold: float = 0.7) -> list[tuple[NodeResult, float]]:
        return await self._entity.find_similar_nodes(embedding, index_name, exclude_ids, limit, threshold)

    async def search_nodes(self, label: str | None = None, search: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return await self._entity.search_nodes(label, search, limit)

    async def find_relationship_by_id(self, rel_id: str) -> dict[str, Any]:
        return await self._entity.find_relationship_by_id(rel_id)

    async def get_node_relationships_detailed(self, node_id: str) -> list[dict[str, Any]]:
        return await self._entity.get_node_relationships_detailed(node_id)

    # ── Cypher ─────────────────────────────────────────────

    async def execute_cypher(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            return await self._client.execute_query(query, parameters)
        except Exception as e:
            raise QueryExecutionError(str(e), query=query) from e

    # ── Schema ─────────────────────────────────────────────

    async def get_schema(self, force_refresh: bool = False) -> dict[str, Any]:
        return await self._schema.get_schema(force_refresh)

    def invalidate_schema_cache(self) -> None:
        self._schema.invalidate_schema_cache()

    async def get_node_labels(self) -> list[str]:
        return await self._schema.get_node_labels()

    async def get_relationship_types(self) -> list[str]:
        return await self._schema.get_relationship_types()

    async def get_node_properties(self, label: str) -> list[str]:
        return await self._schema.get_node_properties(label)

    # ── Vector ─────────────────────────────────────────────

    async def vector_search_nodes(self, embedding: list[float], index_name: str, labels: list[str] | None = None,
                                  limit: int = 10, threshold: float | None = None) -> list[tuple[NodeResult, float]]:
        return await self._vector.vector_search_nodes(embedding, index_name, labels, limit, threshold)

    async def ensure_vector_index(self, index_name: str, label: str, property_name: str, dimensions: int = 1536) -> bool:
        return await self._vector.ensure_vector_index(index_name, label, property_name, dimensions)

    async def upsert_node_embedding(self, node_id: str, property_name: str, embedding: list[float]) -> bool:
        return await self._vector.upsert_node_embedding(node_id, property_name, embedding)

    async def batch_upsert_node_embeddings(self, updates: list[dict[str, Any]], property_name: str) -> int:
        return await self._vector.batch_upsert_node_embeddings(updates, property_name)

    # ── Graph CRUD ─────────────────────────────────────────

    async def create_node_generic(self, label: str, properties: dict[str, Any]) -> dict[str, Any] | None:
        return await self._graph_crud.create_node_generic(label, properties)

    async def check_duplicate_node(self, label: str, name: str) -> bool:
        return await self._graph_crud.check_duplicate_node(label, name)

    async def update_node_properties(self, node_id: str, properties: dict[str, Any],
                                     remove_keys: list[str] | None = None) -> dict[str, Any]:
        return await self._graph_crud.update_node_properties(node_id, properties, remove_keys)

    async def delete_node_generic(self, node_id: str, force: bool = False) -> bool:
        return await self._graph_crud.delete_node_generic(node_id, force)

    async def get_node_relationship_count(self, node_id: str) -> int:
        return await self._graph_crud.get_node_relationship_count(node_id)

    async def delete_node_atomic(self, node_id: str, force: bool = False) -> dict[str, Any]:
        return await self._graph_crud.delete_node_atomic(node_id, force)

    async def create_relationship_generic(self, source_id: str, target_id: str, rel_type: str,
                                          properties: dict[str, Any] | None = None) -> dict[str, Any]:
        return await self._graph_crud.create_relationship_generic(source_id, target_id, rel_type, properties)

    async def delete_relationship_generic(self, rel_id: str) -> bool:
        return await self._graph_crud.delete_relationship_generic(rel_id)
