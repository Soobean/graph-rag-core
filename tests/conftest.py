"""
Test Configuration

테스트 공통 fixture 정의 — 모든 외부 의존성을 mock으로 대체합니다.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Settings


@pytest.fixture
def mock_settings():
    """테스트용 Settings mock"""
    settings = MagicMock(spec=Settings)
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "password"
    settings.neo4j_database = "neo4j"
    settings.neo4j_max_connection_pool_size = 10
    settings.neo4j_connection_timeout = 5.0
    settings.azure_openai_endpoint = "https://test.openai.azure.com/"
    settings.azure_openai_api_key = "test-key"
    settings.azure_openai_deployment = "gpt-4"
    settings.azure_openai_api_version = "2024-02-15-preview"
    settings.app_name = "Graph RAG Core Test"
    settings.app_version = "0.1.0"
    settings.log_level = "DEBUG"
    settings.log_format = "%(message)s"
    settings.cors_origins = ["http://localhost:3000"]
    settings.auth_enabled = False
    settings.is_production = False
    settings.is_development = True
    settings.vector_search_enabled = False
    settings.prompts_dir = "src/prompts"
    settings.checkpointer_db_path = ":memory:"
    return settings


@pytest.fixture
def mock_neo4j():
    """Neo4jRepository mock"""
    repo = AsyncMock()
    repo.get_schema.return_value = {
        "node_labels": ["Person", "Organization"],
        "relationship_types": ["WORKS_AT", "KNOWS"],
        "node_properties": {
            "Person": ["name", "email"],
            "Organization": ["name", "industry"],
        },
        "relationship_properties": {
            "WORKS_AT": ["since"],
            "KNOWS": [],
        },
    }
    repo.get_node_labels.return_value = ["Person", "Organization"]
    repo.get_relationship_types.return_value = ["WORKS_AT", "KNOWS"]
    repo.find_entities_by_name.return_value = []
    repo.execute_cypher.return_value = []
    repo.check_duplicate_node.return_value = False
    return repo


@pytest.fixture
def mock_llm():
    """LLMRepository mock"""
    repo = AsyncMock()
    repo.classify_intent_and_extract_entities.return_value = {
        "intent": "entity_search",
        "confidence": 0.95,
        "entities": [],
    }
    repo.generate_cypher.return_value = {
        "cypher": "MATCH (n:Person) WHERE toLower(n.name) = toLower($name) RETURN n LIMIT 200",
        "parameters": {"name": "test"},
        "explanation": "Find person by name",
    }
    repo.generate_response.return_value = "Test response"
    repo.close = AsyncMock()
    return repo


@pytest.fixture
def mock_neo4j_client():
    """Neo4jClient mock"""
    client = AsyncMock()
    client.execute_query.return_value = ([], None, None)
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_graph_schema():
    """테스트용 그래프 스키마"""
    return {
        "node_labels": ["Person", "Organization"],
        "relationship_types": ["WORKS_AT", "KNOWS"],
        "node_properties": {
            "Person": ["name", "email"],
            "Organization": ["name", "industry"],
        },
        "relationship_properties": {
            "WORKS_AT": ["since"],
            "KNOWS": [],
        },
    }
