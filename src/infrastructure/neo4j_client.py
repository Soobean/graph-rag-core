"""
Neo4j 비동기 드라이버 래퍼

책임:
- Neo4j 드라이버 연결 관리 (연결 풀링)
- 비동기 세션 컨텍스트 제공
- 연결 상태 확인 (health check)
- 리소스 정리 (graceful shutdown)
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse, urlunparse

from neo4j import (
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
    AsyncSession,
    basic_auth,
)
from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
from neo4j.graph import Node, Path, Relationship

from src.domain.exceptions import (
    DatabaseAuthenticationError,
    DatabaseConnectionError,
    DatabaseError,
)
from src.domain.validators import validate_cypher_identifier

logger = logging.getLogger(__name__)


def _serialize_value(value: Any) -> Any:
    """Neo4j 반환값을 JSON 직렬화 가능한 형태로 변환"""
    if value is None:
        return None

    if isinstance(value, Node):
        return {
            "id": value.element_id,
            "elementId": value.element_id,
            "labels": list(value.labels),
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    elif isinstance(value, Relationship):
        return {
            "id": value.element_id,
            "elementId": value.element_id,
            "type": value.type,
            "startNodeId": value.start_node.element_id if value.start_node else None,
            "endNodeId": value.end_node.element_id if value.end_node else None,
            "properties": {k: _serialize_value(v) for k, v in dict(value).items()},
        }
    elif isinstance(value, Path):
        return {
            "nodes": [_serialize_value(node) for node in value.nodes],
            "relationships": [_serialize_value(rel) for rel in value.relationships],
        }
    elif isinstance(value, list):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}

    # Neo4j DateTime/Date/Time/Duration → ISO 문자열
    type_name = type(value).__name__
    if type_name in ("DateTime", "Date", "Time", "Duration"):
        if hasattr(value, "isoformat"):
            return value.isoformat()
        elif hasattr(value, "iso_format"):
            return value.iso_format()
        return str(value)

    return value


def _sanitize_uri(uri: str) -> str:
    """URI에서 비밀번호 제거 (로깅용)"""
    try:
        parsed = urlparse(uri)
        if parsed.password:
            netloc = f"{parsed.username}:***@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            sanitized = parsed._replace(netloc=netloc)
            return urlunparse(sanitized)
    except Exception:
        return uri.split("@")[-1] if "@" in uri else uri
    return uri


class TransactionScope:
    """다중 쿼리 트랜잭션 내에서 쿼리를 실행하기 위한 헬퍼"""

    def __init__(self, tx: Any) -> None:
        self._tx = tx

    async def run_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """트랜잭션 내에서 쿼리 실행 및 직렬화된 결과 반환"""
        result = await self._tx.run(query, parameters or {})
        records = []
        async for record in result:
            serialized = {key: _serialize_value(record[key]) for key in record.keys()}
            records.append(serialized)
        return records


class Neo4jClient:
    """Neo4j 비동기 드라이버 래퍼"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
        schema_excluded_labels: set[str] | None = None,
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._max_connection_pool_size = max_connection_pool_size
        self._connection_timeout = connection_timeout
        self._schema_excluded_labels = schema_excluded_labels or set()

        self._driver: AsyncDriver | None = None

        logger.info(
            f"Neo4jClient initialized: uri={_sanitize_uri(uri)}, database={database}, "
            f"pool_size={max_connection_pool_size}"
        )

    async def connect(self) -> None:
        """Neo4j 드라이버 연결 초기화"""
        if self._driver is not None:
            logger.debug("Driver already connected, skipping connection")
            return

        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=basic_auth(self._user, self._password),
                max_connection_pool_size=self._max_connection_pool_size,
                connection_timeout=self._connection_timeout,
            )
            await self._driver.verify_connectivity()
            logger.info(
                f"Successfully connected to Neo4j at {_sanitize_uri(self._uri)}"
            )

        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise DatabaseAuthenticationError(
                f"Failed to authenticate with Neo4j: {e}"
            ) from e
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise DatabaseConnectionError(
                f"Neo4j service is unavailable at {_sanitize_uri(self._uri)}: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise DatabaseConnectionError(f"Failed to connect to Neo4j: {e}") from e

    async def close(self) -> None:
        """Neo4j 드라이버 연결 종료"""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j driver closed")

    async def __aenter__(self) -> "Neo4jClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        await self.close()

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            raise DatabaseConnectionError(
                "Neo4j driver is not initialized. Call connect() first."
            )
        return self._driver

    @asynccontextmanager
    async def session(
        self,
        database: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AsyncSession]:
        db = database or self._database
        session = self.driver.session(database=db, **kwargs)
        try:
            yield session
        finally:
            await session.close()

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """단일 쿼리 실행 및 결과 반환"""
        try:
            async with self.session(database=database) as session:
                result = await session.run(query, parameters or {})
                records = []
                async for record in result:
                    serialized = {
                        key: _serialize_value(record[key]) for key in record.keys()
                    }
                    records.append(serialized)
                logger.debug(
                    f"Query executed successfully: {len(records)} records returned"
                )
                return records
        except Neo4jError as e:
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"Failed to execute query: {e}") from e

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
    ) -> list[dict[str, Any]]:
        """쓰기 쿼리 실행 (트랜잭션 보장)"""

        async def _write_tx(
            tx: AsyncManagedTransaction, q: str, params: dict[str, Any]
        ) -> list[dict[str, Any]]:
            result = await tx.run(q, params)
            records = []
            async for record in result:
                serialized = {
                    key: _serialize_value(record[key]) for key in record.keys()
                }
                records.append(serialized)
            return records

        try:
            async with self.session(database=database) as session:
                records = await session.execute_write(
                    _write_tx, query, parameters or {}
                )
                logger.debug(f"Write query executed: {len(records)} records affected")
                return records
        except Neo4jError as e:
            logger.error(f"Write query failed: {e}")
            raise DatabaseError(f"Failed to execute write query: {e}") from e

    @asynccontextmanager
    async def begin_transaction(
        self,
        database: str | None = None,
    ) -> AsyncIterator["TransactionScope"]:
        """다중 쿼리 트랜잭션 컨텍스트 매니저"""
        db = database or self._database
        session = self.driver.session(database=db)
        tx = await session.begin_transaction()
        scope = TransactionScope(tx)
        try:
            yield scope
            await tx.commit()
        except Exception:
            if not tx.closed:
                await tx.rollback()
            raise
        finally:
            await session.close()

    async def health_check(self) -> dict[str, Any]:
        """Neo4j 연결 상태 확인"""
        result = {
            "connected": False,
            "uri": _sanitize_uri(self._uri),
            "database": self._database,
            "server_info": None,
            "error": None,
        }

        if self._driver is None:
            result["error"] = "Driver not initialized"
            return result

        try:
            await self._driver.verify_connectivity()
            server_info = await self._driver.get_server_info()
            result["connected"] = True
            result["server_info"] = {
                "address": str(server_info.address),
                "agent": server_info.agent,
                "protocol_version": server_info.protocol_version,
            }
        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"Health check failed: {e}")

        return result

    async def get_schema_info(self) -> dict[str, Any]:
        """데이터베이스 스키마 정보 조회"""
        schema_info: dict[str, Any] = {
            "node_labels": [],
            "relationship_types": [],
            "indexes": [],
            "constraints": [],
        }

        try:
            # 노드 레이블 조회 (excluded labels 제외)
            labels_result = await self.execute_query("CALL db.labels()")
            schema_info["node_labels"] = [
                r.get("label")
                for r in labels_result
                if r.get("label") not in self._schema_excluded_labels
            ]

            # 관계 타입 조회
            rel_result = await self.execute_query("CALL db.relationshipTypes()")
            schema_info["relationship_types"] = [
                r.get("relationshipType") for r in rel_result
            ]

            # 인덱스 조회
            idx_result = await self.execute_query("SHOW INDEXES")
            schema_info["indexes"] = [
                {
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "labelsOrTypes": r.get("labelsOrTypes"),
                    "properties": r.get("properties"),
                }
                for r in idx_result
            ]

            # 제약 조건 조회
            const_result = await self.execute_query("SHOW CONSTRAINTS")
            schema_info["constraints"] = [
                {
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "labelsOrTypes": r.get("labelsOrTypes"),
                    "properties": r.get("properties"),
                }
                for r in const_result
            ]

        except Exception as e:
            logger.warning(f"Failed to get schema info: {e}")
            schema_info["error"] = str(e)

        # 속성 정보 인트로스펙션
        try:
            node_schemas = []
            for label in schema_info.get("node_labels", []):
                prop_result = await self.execute_query(
                    f"MATCH (n:`{label}`) UNWIND keys(n) AS key "
                    "RETURN DISTINCT key LIMIT 50"
                )
                props = [{"name": r["key"]} for r in prop_result if r.get("key")]
                node_schemas.append({"label": label, "properties": props})
            if node_schemas:
                schema_info["nodes"] = node_schemas

            # enum-like 속성 샘플링
            for node_schema in node_schemas:
                label = node_schema["label"]
                for prop in node_schema.get("properties", []):
                    prop_name = prop.get("name", "")
                    if not prop_name or prop_name in ("id", "name", "embedding"):
                        continue
                    try:
                        distinct_result = await self.execute_query(
                            f"MATCH (n:`{label}`) WHERE n.`{prop_name}` IS NOT NULL "
                            f"WITH DISTINCT n.`{prop_name}` AS val "
                            f"WITH val WHERE val IS :: STRING "
                            f"RETURN collect(val)[..20] AS vals, count(val) AS cnt"
                        )
                        if distinct_result:
                            row = distinct_result[0]
                            cnt = row.get("cnt", 0)
                            vals = row.get("vals", [])
                            if 2 <= cnt <= 20 and vals:
                                prop["sample_values"] = vals
                    except Exception:
                        pass

            rel_schemas = []
            for rel_type in schema_info.get("relationship_types", []):
                prop_result = await self.execute_query(
                    f"MATCH ()-[r:`{rel_type}`]->() UNWIND keys(r) AS key "
                    "RETURN DISTINCT key LIMIT 50"
                )
                props = [{"name": r["key"]} for r in prop_result if r.get("key")]
                rel_schemas.append({"type": rel_type, "properties": props})
            # 관계 속성 enum-like 값 샘플링
            for rel_schema in rel_schemas:
                rel_type = rel_schema["type"]
                for prop in rel_schema.get("properties", []):
                    prop_name = prop.get("name", "")
                    if not prop_name or prop_name in ("id", "embedding"):
                        continue
                    try:
                        distinct_result = await self.execute_query(
                            f"MATCH ()-[r:`{rel_type}`]->() WHERE r.`{prop_name}` IS NOT NULL "
                            f"WITH DISTINCT r.`{prop_name}` AS val "
                            f"WITH val WHERE val IS :: STRING "
                            f"RETURN collect(val)[..20] AS vals, count(val) AS cnt"
                        )
                        if distinct_result:
                            row = distinct_result[0]
                            cnt = row.get("cnt", 0)
                            vals = row.get("vals", [])
                            if 2 <= cnt <= 20 and vals:
                                prop["sample_values"] = vals
                    except Exception:
                        pass

            if rel_schemas:
                schema_info["relationships"] = rel_schemas
        except Exception as e:
            logger.warning(f"Failed to introspect properties: {e}")

        return schema_info

    # ============================================
    # Vector Index 관련 메서드
    # ============================================

    async def create_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int = 1536,
        similarity_function: str = "cosine",
    ) -> bool:
        """Neo4j Vector Index 생성"""
        safe_index = validate_cypher_identifier(index_name, "index_name")
        safe_label = validate_cypher_identifier(label, "label")
        safe_prop = validate_cypher_identifier(property_name, "property_name")

        check_query = """
        SHOW INDEXES
        WHERE name = $index_name
        """
        existing = await self.execute_query(check_query, {"index_name": safe_index})
        if existing:
            logger.info(f"Vector index '{safe_index}' already exists")
            return True

        create_query = f"""
        CREATE VECTOR INDEX `{safe_index}` IF NOT EXISTS
        FOR (n:`{safe_label}`)
        ON (n.`{safe_prop}`)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: $dimensions,
                `vector.similarity_function`: $similarity_function
            }}
        }}
        """
        try:
            await self.execute_write(
                create_query,
                {"dimensions": dimensions, "similarity_function": similarity_function},
            )
            logger.info(
                f"Created vector index '{index_name}' on :{label}.{property_name} "
                f"(dims={dimensions}, similarity={similarity_function})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create vector index '{index_name}': {e}")
            raise DatabaseError(f"Failed to create vector index: {e}") from e

    async def drop_vector_index(self, index_name: str) -> bool:
        """Vector Index 삭제"""
        safe_index = validate_cypher_identifier(index_name, "index_name")
        drop_query = f"DROP INDEX `{safe_index}` IF EXISTS"
        try:
            await self.execute_write(drop_query)
            logger.info(f"Dropped vector index '{safe_index}'")
            return True
        except Exception as e:
            logger.error(f"Failed to drop vector index '{index_name}': {e}")
            raise DatabaseError(f"Failed to drop vector index: {e}") from e

    async def vector_search(
        self,
        index_name: str,
        embedding: list[float],
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Vector Index를 사용한 유사도 검색"""
        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        """

        if threshold is not None:
            query += "WHERE score >= $threshold\n"

        query += """
        RETURN elementId(node) as id,
               labels(node) as labels,
               properties(node) as properties,
               score
        ORDER BY score DESC
        """

        params: dict[str, Any] = {
            "index_name": index_name,
            "limit": limit,
            "embedding": embedding,
        }
        if threshold is not None:
            params["threshold"] = threshold

        try:
            results = await self.execute_query(query, params)
            logger.debug(
                f"Vector search on '{index_name}': {len(results)} results (limit={limit})"
            )
            return results
        except Exception as e:
            logger.error(f"Vector search failed on '{index_name}': {e}")
            raise DatabaseError(f"Vector search failed: {e}") from e

    async def upsert_embedding(
        self,
        node_id: str,
        property_name: str,
        embedding: list[float],
    ) -> bool:
        """노드에 임베딩 벡터 저장/업데이트"""
        safe_prop = validate_cypher_identifier(property_name, "property_name")

        query = f"""
        MATCH (n)
        WHERE elementId(n) = $node_id
        SET n.`{safe_prop}` = $embedding
        RETURN elementId(n) as id
        """
        try:
            result = await self.execute_write(
                query, {"node_id": node_id, "embedding": embedding}
            )
            if result:
                logger.debug(f"Upserted embedding on node {node_id}")
                return True
            logger.warning(f"Node {node_id} not found for embedding upsert")
            return False
        except Exception as e:
            logger.error(f"Failed to upsert embedding on node {node_id}: {e}")
            raise DatabaseError(f"Failed to upsert embedding: {e}") from e

    async def batch_upsert_embeddings(
        self,
        updates: list[dict[str, Any]],
        property_name: str,
    ) -> int:
        """여러 노드에 임베딩 일괄 저장"""
        safe_prop = validate_cypher_identifier(property_name, "property_name")

        query = f"""
        UNWIND $updates AS update
        MATCH (n)
        WHERE elementId(n) = update.node_id
        SET n.`{safe_prop}` = update.embedding
        RETURN count(n) as updated_count
        """
        try:
            result = await self.execute_write(query, {"updates": updates})
            count = result[0]["updated_count"] if result else 0
            logger.info(f"Batch upserted embeddings: {count} nodes updated")
            return count
        except Exception as e:
            logger.error(f"Batch embedding upsert failed: {e}")
            raise DatabaseError(f"Batch embedding upsert failed: {e}") from e
