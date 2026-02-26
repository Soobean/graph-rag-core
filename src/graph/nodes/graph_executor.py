"""
Graph Executor Node — Cypher 실행

캐시 저장은 Cypher 실행 성공 후에만 수행합니다 (실패한 Cypher가 캐시되는 것을 방지).
"""

from typing import Any

from src.config import Settings
from src.domain.types import GraphExecutorUpdate
from src.domain.validators import validate_read_only_cypher
from src.graph.nodes.base import DB_TIMEOUT, BaseNode
from src.graph.state import GraphRAGState
from src.repositories.neo4j_repository import Neo4jRepository
from src.repositories.query_cache_repository import QueryCacheRepository


class GraphExecutorNode(BaseNode[GraphExecutorUpdate]):
    """그래프 쿼리 실행 노드"""

    def __init__(
        self,
        neo4j_repository: Neo4jRepository,
        cache_repository: QueryCacheRepository | None = None,
        settings: Settings | None = None,
    ):
        super().__init__()
        self._neo4j = neo4j_repository
        self._cache = cache_repository
        self._settings = settings

    @property
    def name(self) -> str:
        return "graph_executor"

    @property
    def timeout_seconds(self) -> float:
        return DB_TIMEOUT

    @property
    def input_keys(self) -> list[str]:
        return ["cypher_query"]

    async def _process(self, state: GraphRAGState) -> GraphExecutorUpdate:
        """Cypher 쿼리 실행"""
        cypher_query = state.get("cypher_query", "")
        parameters = state.get("cypher_parameters", {})

        if not cypher_query:
            self._logger.warning("No Cypher query to execute")
            return GraphExecutorUpdate(
                graph_results=[],
                result_count=0,
                execution_path=[f"{self.name}_skipped"],
            )

        self._logger.info(f"Executing Cypher: {cypher_query[:100]}...")

        try:
            validate_read_only_cypher(cypher_query)
        except ValueError as e:
            self._logger.error(f"Write query blocked: {e}")
            return GraphExecutorUpdate(
                graph_results=[],
                result_count=0,
                error=f"Security: {e}",
                execution_path=[f"{self.name}_blocked"],
            )

        try:
            results = await self._neo4j.execute_cypher(
                query=cypher_query,
                parameters=parameters,
            )

            self._logger.info(f"Query returned {len(results)} results")

            # 실행 성공 시에만 캐시 저장 (실패한 Cypher 캐시 방지)
            if results and not state.get("cache_hit"):
                await self._save_to_cache(state, cypher_query, parameters)

            return GraphExecutorUpdate(
                graph_results=results,
                result_count=len(results),
                execution_path=[self.name],
            )

        except Exception as e:
            self._logger.error(f"Query execution failed: {e}")
            return GraphExecutorUpdate(
                graph_results=[],
                result_count=0,
                error=f"Query execution failed: {str(e)}",
                execution_path=[f"{self.name}_error"],
            )

    async def _save_to_cache(
        self,
        state: GraphRAGState,
        cypher: str,
        parameters: dict[str, Any],
    ) -> None:
        """실행 성공한 Cypher 쿼리를 캐시에 저장"""
        if not self._cache:
            return
        if not self._settings or not self._settings.vector_search_enabled:
            return

        embedding = state.get("question_embedding")
        if not embedding:
            return

        try:
            question = state.get("question", "")
            await self._cache.cache_query(
                question=question,
                embedding=embedding,
                cypher_query=cypher,
                cypher_parameters=parameters,
            )
            self._logger.debug(f"Cached successful query: {question[:50]}...")
        except Exception as e:
            self._logger.warning(f"Failed to save query to cache: {e}")
