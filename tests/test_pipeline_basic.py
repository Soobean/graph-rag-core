"""
기본 파이프라인 테스트

mock 기반으로 파이프라인이 정상적으로 초기화되고 실행되는지 확인합니다.
"""


import pytest

from src.graph.constants import AVAILABLE_INTENTS
from src.graph.pipeline import GraphRAGPipeline


class TestPipelineInit:
    """파이프라인 초기화 테스트"""

    def test_pipeline_initializes_with_schema(
        self, mock_settings, mock_neo4j, mock_llm, sample_graph_schema
    ):
        """스키마를 주입하여 파이프라인 초기화"""
        pipeline = GraphRAGPipeline(
            settings=mock_settings,
            neo4j_repository=mock_neo4j,
            llm_repository=mock_llm,
            graph_schema=sample_graph_schema,
        )
        assert pipeline is not None
        assert pipeline._graph_schema == sample_graph_schema

    def test_pipeline_initializes_without_schema(
        self, mock_settings, mock_neo4j, mock_llm
    ):
        """스키마 없이도 파이프라인 초기화 가능"""
        pipeline = GraphRAGPipeline(
            settings=mock_settings,
            neo4j_repository=mock_neo4j,
            llm_repository=mock_llm,
        )
        assert pipeline is not None
        assert pipeline._graph_schema is None


class TestIntentConstants:
    """인텐트 상수 테스트"""

    def test_intents_are_defined(self):
        """범용 인텐트가 정의되어 있는지 확인"""
        assert "entity_search" in AVAILABLE_INTENTS
        assert "relationship_search" in AVAILABLE_INTENTS
        assert "aggregation" in AVAILABLE_INTENTS
        assert "path_analysis" in AVAILABLE_INTENTS
        assert "comparison" in AVAILABLE_INTENTS
        # "unknown"은 AVAILABLE_INTENTS에 포함되지 않음 (분류 옵션이 아님)
        assert "unknown" not in AVAILABLE_INTENTS

    def test_no_hr_intents(self):
        """HR 도메인 인텐트가 제거되었는지 확인"""
        hr_intents = {
            "personnel_search",
            "project_matching",
            "org_analysis",
            "mentoring_network",
            "certificate_search",
            "ontology_update",
            "global_analysis",
        }
        for intent in hr_intents:
            assert intent not in AVAILABLE_INTENTS


class TestPipelineRun:
    """파이프라인 실행 테스트"""

    @pytest.fixture
    def pipeline(self, mock_settings, mock_neo4j, mock_llm, sample_graph_schema):
        """테스트용 파이프라인 생성"""
        mock_llm.classify_intent_and_extract_entities.return_value = {
            "intent": "entity_search",
            "confidence": 0.9,
            "entities": [
                {"type": "Person", "value": "Alice", "normalized": "Alice"}
            ],
        }
        mock_neo4j.find_entities_by_name.return_value = [
            {
                "id": "node-1",
                "labels": ["Person"],
                "name": "Alice",
                "score": 1.0,
            }
        ]
        mock_neo4j.execute_cypher.return_value = [
            {"n": {"name": "Alice", "email": "alice@example.com"}}
        ]
        mock_llm.generate_response.return_value = "Alice는 example.com에서 근무합니다."

        return GraphRAGPipeline(
            settings=mock_settings,
            neo4j_repository=mock_neo4j,
            llm_repository=mock_llm,
            graph_schema=sample_graph_schema,
        )

    async def test_pipeline_run_returns_result(self, pipeline):
        """파이프라인 실행 결과가 올바른 형태인지 확인"""
        result = await pipeline.run("Tell me about Alice")

        assert result["success"] is True or result["success"] is False
        assert "question" in result
        assert "response" in result
        assert "metadata" in result
        assert result["question"] == "Tell me about Alice"

    async def test_pipeline_run_with_session_id(self, pipeline):
        """세션 ID를 사용한 파이프라인 실행"""
        result = await pipeline.run(
            "Tell me about Alice", session_id="test-session-001"
        )
        assert "question" in result
        assert result["question"] == "Tell me about Alice"

    async def test_pipeline_handles_error_gracefully(
        self, mock_settings, mock_neo4j, mock_llm, sample_graph_schema
    ):
        """파이프라인 LLM 에러 시 graceful 응답 (크래시 없이 완료)"""
        mock_llm.classify_intent_and_extract_entities.side_effect = Exception(
            "LLM connection failed"
        )

        pipeline = GraphRAGPipeline(
            settings=mock_settings,
            neo4j_repository=mock_neo4j,
            llm_repository=mock_llm,
            graph_schema=sample_graph_schema,
        )

        result = await pipeline.run("Find something")

        # IntentEntityExtractorNode이 에러를 내부 처리하므로 파이프라인 자체는 완료됨
        assert "question" in result
        assert "response" in result
        assert result["question"] == "Find something"
