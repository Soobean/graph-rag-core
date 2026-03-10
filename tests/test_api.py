"""
API Routes 테스트

query.py, graph_edit.py 라우터의 Happy path / Edge case / Error case를 검증합니다.

전략:
- lifespan을 건너뛰고 app.state에 mock을 직접 주입하는 test_app fixture 사용
- get_current_user는 dependency_overrides로 UserContext.anonymous_admin() 반환
- 모든 외부 의존성(pipeline, neo4j_client, graph_edit_service 등)은 AsyncMock/MagicMock
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import graph_edit_router, query_router
from src.auth.models import UserContext
from src.dependencies import get_current_user
from src.domain.exceptions import EntityNotFoundError, LLMRateLimitError, ValidationError


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def mock_pipeline() -> AsyncMock:
    pipeline = AsyncMock()
    pipeline.run.return_value = {
        "success": True,
        "question": "Who knows Alice?",
        "response": "Bob knows Alice.",
        "metadata": {
            "intent": "entity_search",
            "intent_confidence": 0.95,
            "entities": {},
            "resolved_entities": [],
            "cypher_query": "MATCH (n) RETURN n",
            "cypher_parameters": {},
            "result_count": 1,
            "execution_path": ["intent_entity_extractor", "response_generator"],
            "error": None,
        },
        "error": None,
    }
    return pipeline


@pytest.fixture
def mock_neo4j_client() -> AsyncMock:
    client = AsyncMock()
    client.health_check.return_value = {
        "connected": True,
        "server_info": {"version": "5.x"},
    }
    client.get_schema_info.return_value = {
        "node_labels": ["Person", "Organization"],
        "relationship_types": ["KNOWS", "WORKS_AT"],
        "indexes": [],
        "constraints": [],
    }
    return client


@pytest.fixture
def mock_graph_edit_service() -> AsyncMock:
    service = AsyncMock()
    service.create_node.return_value = {
        "id": "node-1",
        "labels": ["Person"],
        "properties": {"name": "Alice"},
    }
    service.search_nodes.return_value = [
        {"id": "node-1", "labels": ["Person"], "properties": {"name": "Alice"}}
    ]
    service.delete_node.return_value = None
    service.create_edge.return_value = {
        "id": "edge-1",
        "type": "KNOWS",
        "source_id": "node-1",
        "target_id": "node-2",
        "properties": {},
    }
    service.get_schema_info.return_value = {
        "allowed_labels": ["Person", "Organization"],
        "required_properties": {"Person": ["name"]},
        "valid_relationships": ["KNOWS", "WORKS_AT"],
    }
    return service


@pytest.fixture
def mock_explainability_service() -> MagicMock:
    service = MagicMock()
    service.build_thought_process.return_value = None
    service.build_graph_data.return_value = None
    return service


@pytest.fixture
def test_app(
    mock_pipeline: AsyncMock,
    mock_neo4j_client: AsyncMock,
    mock_graph_edit_service: AsyncMock,
    mock_explainability_service: MagicMock,
) -> FastAPI:
    """
    lifespan 없이 app.state에 mock을 직접 주입한 테스트용 FastAPI 앱.
    get_current_user는 dependency_overrides로 anonymous_admin을 반환합니다.
    """
    app = FastAPI()
    app.include_router(query_router)
    app.include_router(graph_edit_router)

    # app.state에 mock 주입 (lifespan 대체)
    app.state.pipeline = mock_pipeline
    app.state.neo4j_client = mock_neo4j_client
    app.state.neo4j_repo = AsyncMock()
    app.state.llm_repo = AsyncMock()
    app.state.graph_edit_service = mock_graph_edit_service
    app.state.explainability_service = mock_explainability_service
    app.state.auth_service = AsyncMock()

    # 인증 우회
    app.dependency_overrides[get_current_user] = lambda: UserContext.anonymous_admin()

    return app


@pytest.fixture
async def client(test_app: FastAPI) -> AsyncClient:
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac


# ============================================
# Query Routes 테스트
# ============================================


class TestQueryEndpoint:
    async def test_query_success(self, client: AsyncClient) -> None:
        """POST /api/v1/query - 정상 응답"""
        resp = await client.post(
            "/api/v1/query", json={"question": "Who knows Alice?"}
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["question"] == "Who knows Alice?"
        assert data["response"] == "Bob knows Alice."

    async def test_query_empty_question_returns_422(self, client: AsyncClient) -> None:
        """POST /api/v1/query - 빈 question은 422 Unprocessable Entity"""
        resp = await client.post("/api/v1/query", json={"question": ""})

        assert resp.status_code == 422

    async def test_query_pipeline_error_returns_500(
        self, client: AsyncClient, mock_pipeline: AsyncMock
    ) -> None:
        """POST /api/v1/query - 파이프라인 예외 시 500"""
        mock_pipeline.run.side_effect = RuntimeError("unexpected failure")

        resp = await client.post(
            "/api/v1/query", json={"question": "Who knows Alice?"}
        )

        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    async def test_query_llm_rate_limit_returns_429(
        self, client: AsyncClient, mock_pipeline: AsyncMock
    ) -> None:
        """POST /api/v1/query - LLMRateLimitError 시 429"""
        mock_pipeline.run.side_effect = LLMRateLimitError("rate limit exceeded", retry_after=30)

        resp = await client.post(
            "/api/v1/query", json={"question": "Who knows Alice?"}
        )

        assert resp.status_code == 429
        assert resp.headers.get("retry-after") == "30"

    async def test_query_domain_validation_error_returns_400(
        self, client: AsyncClient, mock_pipeline: AsyncMock
    ) -> None:
        """POST /api/v1/query - DomainValidationError 시 400"""
        mock_pipeline.run.side_effect = ValidationError("invalid input", field="question")

        resp = await client.post(
            "/api/v1/query", json={"question": "bad query"}
        )

        assert resp.status_code == 400


class TestQueryStreamEndpoint:
    async def test_stream_returns_sse_events(
        self, client: AsyncClient, mock_pipeline: AsyncMock
    ) -> None:
        """POST /api/v1/query/stream - SSE 이벤트 스트리밍"""

        # run_with_streaming_response는 async generator이므로
        # AsyncMock이 아닌 일반 MagicMock으로 교체하여 async generator를 직접 반환합니다.
        async def fake_stream(*args, **kwargs):
            yield {"type": "step", "data": {"node_name": "intent_entity_extractor", "description": "Extracting intent", "step_number": 1}}
            yield {"type": "chunk", "text": "Bob knows"}
            yield {"type": "chunk", "text": " Alice."}
            yield {"type": "done", "success": True, "full_response": "Bob knows Alice."}

        mock_pipeline.run_with_streaming_response = fake_stream

        resp = await client.post(
            "/api/v1/query/stream", json={"question": "Who knows Alice?"}
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        body = resp.text
        assert "event: step" in body
        assert "event: chunk" in body
        assert "event: done" in body


class TestHealthEndpoint:
    async def test_health_returns_healthy(self, client: AsyncClient) -> None:
        """GET /api/v1/health - Neo4j 연결 정상 시 healthy"""
        resp = await client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["neo4j_connected"] is True

    async def test_health_returns_degraded_when_neo4j_disconnected(
        self, client: AsyncClient, mock_neo4j_client: AsyncMock
    ) -> None:
        """GET /api/v1/health - Neo4j 연결 실패 시 degraded"""
        mock_neo4j_client.health_check.return_value = {
            "connected": False,
            "server_info": None,
        }

        resp = await client.get("/api/v1/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["neo4j_connected"] is False


class TestSchemaEndpoint:
    async def test_schema_returns_graph_schema(self, client: AsyncClient) -> None:
        """GET /api/v1/schema - 스키마 데이터 반환"""
        resp = await client.get("/api/v1/schema")

        assert resp.status_code == 200
        data = resp.json()
        assert "Person" in data["node_labels"]
        assert "KNOWS" in data["relationship_types"]
        assert isinstance(data["indexes"], list)
        assert isinstance(data["constraints"], list)


# ============================================
# Graph Edit Routes 테스트
# ============================================


class TestCreateNodeEndpoint:
    async def test_create_node_success_returns_201(self, client: AsyncClient) -> None:
        """POST /api/v1/graph/nodes - 노드 생성 성공 시 201"""
        resp = await client.post(
            "/api/v1/graph/nodes",
            json={"label": "Person", "properties": {"name": "Alice"}},
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "node-1"
        assert "Person" in data["labels"]
        assert data["properties"]["name"] == "Alice"

    async def test_create_node_validation_error_returns_400(
        self, client: AsyncClient, mock_graph_edit_service: AsyncMock
    ) -> None:
        """POST /api/v1/graph/nodes - ValidationError 시 400"""
        mock_graph_edit_service.create_node.side_effect = ValidationError(
            "name property is required", field="name"
        )

        resp = await client.post(
            "/api/v1/graph/nodes",
            json={"label": "Person", "properties": {}},
        )

        assert resp.status_code == 400


class TestSearchNodesEndpoint:
    async def test_search_nodes_success(self, client: AsyncClient) -> None:
        """GET /api/v1/graph/nodes - 노드 검색 성공"""
        resp = await client.get("/api/v1/graph/nodes", params={"label": "Person"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["nodes"][0]["id"] == "node-1"

    async def test_search_nodes_empty_result(
        self, client: AsyncClient, mock_graph_edit_service: AsyncMock
    ) -> None:
        """GET /api/v1/graph/nodes - 결과 없음"""
        mock_graph_edit_service.search_nodes.return_value = []

        resp = await client.get("/api/v1/graph/nodes", params={"search": "nonexistent"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["nodes"] == []


class TestDeleteNodeEndpoint:
    async def test_delete_node_success_returns_204(self, client: AsyncClient) -> None:
        """DELETE /api/v1/graph/nodes/{id} - 삭제 성공 시 204"""
        resp = await client.delete("/api/v1/graph/nodes/node-1")

        assert resp.status_code == 204

    async def test_delete_node_not_found_returns_404(
        self, client: AsyncClient, mock_graph_edit_service: AsyncMock
    ) -> None:
        """DELETE /api/v1/graph/nodes/{id} - 존재하지 않는 노드 시 404"""
        mock_graph_edit_service.delete_node.side_effect = EntityNotFoundError(
            "Node", "node-999"
        )

        resp = await client.delete("/api/v1/graph/nodes/node-999")

        assert resp.status_code == 404


class TestCreateEdgeEndpoint:
    async def test_create_edge_success_returns_201(self, client: AsyncClient) -> None:
        """POST /api/v1/graph/edges - 엣지 생성 성공 시 201"""
        resp = await client.post(
            "/api/v1/graph/edges",
            json={
                "source_id": "node-1",
                "target_id": "node-2",
                "relationship_type": "KNOWS",
            },
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "edge-1"
        assert data["type"] == "KNOWS"
        assert data["source_id"] == "node-1"
        assert data["target_id"] == "node-2"

    async def test_create_edge_source_not_found_returns_404(
        self, client: AsyncClient, mock_graph_edit_service: AsyncMock
    ) -> None:
        """POST /api/v1/graph/edges - 소스 노드 없음 시 404"""
        mock_graph_edit_service.create_edge.side_effect = EntityNotFoundError(
            "Node", "node-999"
        )

        resp = await client.post(
            "/api/v1/graph/edges",
            json={
                "source_id": "node-999",
                "target_id": "node-2",
                "relationship_type": "KNOWS",
            },
        )

        assert resp.status_code == 404
