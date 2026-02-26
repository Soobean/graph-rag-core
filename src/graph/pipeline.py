"""
Graph RAG Pipeline

LangGraph를 사용한 RAG 파이프라인 정의
- Checkpointer로 대화 기록 관리
- 스키마는 초기화 시 주입 (런타임 조회 제거)

핵심 6노드 파이프라인:
  IntentEntityExtractor → EntityResolver → CypherGenerator
  → GraphExecutor → ResponseGenerator
  (unresolved → ClarificationHandler)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.auth.models import UserContext
from src.config import Settings
from src.domain.types import GraphSchema, PipelineMetadata, PipelineResult
from src.graph.constants import META_NODE_PREFIXES
from src.graph.metadata_builder import ResponseMetadataBuilder
from src.graph.nodes import (
    ClarificationHandlerNode,
    CypherGeneratorNode,
    EntityResolverNode,
    GraphExecutorNode,
    IntentEntityExtractorNode,
    ResponseGeneratorNode,
)
from src.graph.state import AGGREGATE_INTENTS, GraphRAGState
from src.graph.utils import format_chat_history
from src.infrastructure.neo4j_client import Neo4jClient
from src.repositories.llm_repository import LLMRepository
from src.repositories.neo4j_repository import Neo4jRepository
from src.repositories.query_cache_repository import QueryCacheRepository

logger = logging.getLogger(__name__)


def _build_step_description(
    node_name: str,
    state: dict[str, Any],
    node_output: dict[str, Any],
) -> str:
    """각 파이프라인 노드의 실시간 설명을 생성한다."""
    if node_name == "intent_entity_extractor":
        intent = state.get("intent", "unknown")
        confidence = state.get("intent_confidence", 0)
        return f"의도: {intent} ({confidence * 100:.0f}%)"

    if node_name == "entity_resolver":
        return "그래프에서 엔티티 매칭"

    if node_name == "cypher_generator":
        return "Cypher 쿼리 생성"

    if node_name == "graph_executor":
        results = state.get("graph_results", [])
        count = len(results) if isinstance(results, list) else 0
        return f"쿼리 실행: {count}건 조회"

    if node_name == "response_generator":
        return "응답 생성"

    if node_name == "clarification_handler":
        return "명확화 요청"

    return node_name


class GraphRAGPipeline:
    """
    Graph RAG 파이프라인

    질문 → 의도분류+엔티티추출 → 엔티티해석 → Cypher생성 → 쿼리실행 → 응답생성

    사용 예시:
        schema = await neo4j_repo.get_schema()
        pipeline = GraphRAGPipeline(
            settings=settings,
            neo4j_repository=neo4j_repo,
            llm_repository=llm_repo,
            graph_schema=schema,
        )
        result = await pipeline.run("Find people who know Python", session_id="user-123")
    """

    def __init__(
        self,
        settings: Settings,
        neo4j_repository: Neo4jRepository,
        llm_repository: LLMRepository,
        neo4j_client: Neo4jClient | None = None,
        graph_schema: GraphSchema | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ):
        self._settings = settings
        self._neo4j = neo4j_repository
        self._llm = llm_repository
        self._graph_schema = graph_schema

        # 스키마에서 entity_types와 schema_summary 자동 추출
        self._entity_types = self._extract_entity_types(graph_schema)
        self._schema_summary = self._build_schema_summary(graph_schema)

        # Query Cache Repository (Vector Search 활성화 시)
        self._cache_repository: QueryCacheRepository | None = None
        if settings.vector_search_enabled and neo4j_client:
            self._cache_repository = QueryCacheRepository(neo4j_client, settings)
            logger.info("Query cache repository initialized")

        # 노드 초기화
        self._intent_entity_extractor = IntentEntityExtractorNode(
            llm_repository,
            entity_types=self._entity_types,
            schema_summary=self._schema_summary,
        )
        self._entity_resolver = EntityResolverNode(neo4j_repository)
        self._clarification_handler = ClarificationHandlerNode(llm_repository)
        self._cypher_generator = CypherGeneratorNode(
            llm_repository,
            neo4j_repository,
            settings=settings,
        )
        self._graph_executor = GraphExecutorNode(
            neo4j_repository,
            cache_repository=self._cache_repository,
            settings=settings,
        )
        self._response_generator = ResponseGeneratorNode(llm_repository)

        # 메타데이터 빌더
        self._metadata_builder = ResponseMetadataBuilder()

        # Checkpointer (외부 주입 또는 기본 MemorySaver)
        self._checkpointer = checkpointer or MemorySaver()

        # 그래프 빌드
        self._graph = self._build_graph()

        if not self._entity_types:
            logger.warning(
                "No entity types extracted from schema. "
                "Entity extraction may produce poor results. "
                "Ensure Neo4j has node data."
            )

        logger.info(
            "GraphRAGPipeline initialized "
            f"(schema injected: {graph_schema is not None}, "
            f"entity_types: {self._entity_types}, "
            f"checkpointer: {type(self._checkpointer).__name__})"
        )

    @staticmethod
    def _extract_entity_types(schema: GraphSchema | None) -> list[str]:
        """
        스키마의 node_labels에서 메타 노드를 필터링하여 entity_types를 추출한다.

        메타 노드 기준: '_' 또는 '__' 접두사 (e.g., __Chunk__, _Schema)
        node_labels가 비어있으면 nodes 상세 정보에서 label을 추출한다 (fallback).
        """
        if not schema:
            return []

        node_labels = schema.get("node_labels", [])

        # node_labels가 비어있으면 nodes 상세에서 label 추출 (fallback)
        if not node_labels:
            nodes = schema.get("nodes", [])
            node_labels = [n.get("label", "") for n in nodes if n.get("label")]

        return [
            label
            for label in node_labels
            if not any(label.startswith(prefix) for prefix in META_NODE_PREFIXES)
        ]

    @staticmethod
    def _build_schema_summary(schema: GraphSchema | None) -> str:
        """
        스키마를 사람이 읽을 수 있는 요약 텍스트로 변환한다.

        예: "Node types: Person (name, age, email), Company (name, industry)
             Relationships: WORKS_AT (Person→Company), KNOWS (Person→Person)"
        """
        if not schema:
            return ""

        lines: list[str] = []

        # Node types with properties
        nodes = schema.get("nodes", [])
        if nodes:
            node_parts: list[str] = []
            for node in nodes:
                label = node.get("label", "")
                if not label or any(
                    label.startswith(p) for p in META_NODE_PREFIXES
                ):
                    continue
                props = node.get("properties", [])
                prop_names = [p.get("name", "") for p in props if p.get("name")]
                if prop_names:
                    node_parts.append(f"{label} ({', '.join(prop_names)})")
                else:
                    node_parts.append(label)
            if node_parts:
                lines.append(f"Node types: {', '.join(node_parts)}")
        else:
            # nodes 상세 정보 없으면 node_labels + node_properties fallback
            node_labels = schema.get("node_labels", [])
            filtered = [
                label for label in node_labels
                if not any(label.startswith(p) for p in META_NODE_PREFIXES)
            ]
            if filtered:
                lines.append(f"Node types: {', '.join(filtered)}")

        # Relationships with direction
        rels = schema.get("relationships", [])
        if rels:
            rel_parts: list[str] = []
            for rel in rels:
                rel_type = rel.get("type", "")
                if not rel_type:
                    continue
                start = rel.get("start_labels", [])
                end = rel.get("end_labels", [])
                if start and end:
                    start_str = "/".join(start)
                    end_str = "/".join(end)
                    rel_parts.append(f"{rel_type} ({start_str}→{end_str})")
                else:
                    rel_parts.append(rel_type)
            if rel_parts:
                lines.append(f"Relationships: {', '.join(rel_parts)}")
        else:
            rel_types = schema.get("relationship_types", [])
            if rel_types:
                lines.append(f"Relationships: {', '.join(rel_types)}")

        return "\n".join(lines)

    def _build_graph(self) -> CompiledStateGraph:
        """
        LangGraph 워크플로우 구성

        파이프라인 흐름:
            intent_entity_extractor → entity_resolver
            → cypher_generator → graph_executor → response_generator
        """
        workflow = StateGraph(GraphRAGState)

        # 노드 추가
        workflow.add_node("intent_entity_extractor", self._intent_entity_extractor)
        workflow.add_node("entity_resolver", self._entity_resolver)
        workflow.add_node("clarification_handler", self._clarification_handler)
        workflow.add_node("cypher_generator", self._cypher_generator)
        workflow.add_node("graph_executor", self._graph_executor)
        workflow.add_node("response_generator", self._response_generator)

        # 시작점 설정
        workflow.set_entry_point("intent_entity_extractor")

        # 1. IntentEntityExtractor → EntityResolver 또는 ResponseGenerator
        def route_after_intent(
            state: GraphRAGState,
        ) -> Literal["entity_resolver", "response_generator"]:
            intent = state.get("intent")
            if intent == "unknown":
                logger.info("Intent is unknown. Skipping to response generator.")
                return "response_generator"
            return "entity_resolver"

        workflow.add_conditional_edges(
            "intent_entity_extractor",
            route_after_intent,
            ["entity_resolver", "response_generator"],
        )

        # 2. EntityResolver → CypherGenerator 또는 ClarificationHandler
        def route_after_resolver(
            state: GraphRAGState,
        ) -> Literal["cypher_generator", "clarification_handler", "response_generator"]:
            if state.get("error"):
                return "response_generator"

            unresolved_entities = state.get("unresolved_entities", [])
            resolved_entities = state.get("resolved_entities", [])

            if not unresolved_entities:
                return "cypher_generator"

            if resolved_entities:
                logger.info(
                    f"{len(unresolved_entities)} unresolved entities, "
                    f"but {len(resolved_entities)} resolved. Proceeding to cypher_generator."
                )
                return "cypher_generator"

            intent = state.get("intent", "")
            if intent in AGGREGATE_INTENTS:
                logger.info(
                    f"Aggregate intent '{intent}' with {len(unresolved_entities)} "
                    "unresolved entities. Proceeding to cypher_generator anyway."
                )
                return "cypher_generator"

            entities = state.get("entities", {})
            if entities:
                logger.info(
                    f"All {len(unresolved_entities)} entities unresolved. "
                    "Routing to clarification_handler."
                )
                return "clarification_handler"

            logger.info("No entities found. Routing to clarification_handler.")
            return "clarification_handler"

        workflow.add_conditional_edges(
            "entity_resolver",
            route_after_resolver,
            {
                "cypher_generator": "cypher_generator",
                "clarification_handler": "clarification_handler",
                "response_generator": "response_generator",
            },
        )

        # 3. CypherGenerator → GraphExecutor 또는 에러 처리
        def route_after_cypher(
            state: GraphRAGState,
        ) -> Literal["graph_executor", "response_generator"]:
            if state.get("error") or not state.get("cypher_query"):
                logger.warning("Cypher generation failed. Skipping execution.")
                return "response_generator"
            return "graph_executor"

        workflow.add_conditional_edges(
            "cypher_generator",
            route_after_cypher,
            {
                "graph_executor": "graph_executor",
                "response_generator": "response_generator",
            },
        )

        # 4. GraphExecutor → ResponseGenerator
        workflow.add_edge("graph_executor", "response_generator")

        # 5. ClarificationHandler → END
        workflow.add_edge("clarification_handler", END)

        # 6. ResponseGenerator → END
        workflow.add_edge("response_generator", END)

        return workflow.compile(checkpointer=self._checkpointer)

    async def run(
        self,
        question: str,
        session_id: str | None = None,
        return_full_state: bool = False,
        user_context: UserContext | None = None,
    ) -> PipelineResult:
        """파이프라인 실행"""
        logger.info(f"Running pipeline for: {question[:50]}...")

        thread_id = session_id or str(uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        initial_state: GraphRAGState = {
            "question": question,
            "session_id": session_id or "",
            "messages": [HumanMessage(content=question)],
            "execution_path": [],
        }
        if self._graph_schema:
            initial_state["schema"] = self._graph_schema
        if user_context is not None:
            initial_state["user_context"] = user_context

        try:
            final_state = await self._graph.ainvoke(initial_state, config=config)

            response_text = final_state.get("response", "")

            entities_for_metadata = final_state.get("entities", {})

            query_plan_raw = final_state.get("query_plan")
            query_plan_dict = dict(query_plan_raw) if query_plan_raw else None

            metadata: PipelineMetadata = {
                "intent": final_state.get("intent", "unknown"),
                "intent_confidence": final_state.get("intent_confidence", 0.0),
                "entities": entities_for_metadata,
                "resolved_entities": final_state.get("resolved_entities", []),
                "cypher_query": final_state.get("cypher_query", ""),
                "cypher_parameters": final_state.get("cypher_parameters", {}),
                "result_count": final_state.get("result_count", 0),
                "execution_path": final_state.get("execution_path", []),
                "query_plan": query_plan_dict,
                "error": final_state.get("error"),
            }

            if return_full_state:
                metadata["_full_state"] = {
                    "graph_results": final_state.get("graph_results", []),
                }

            return {
                "success": True,
                "question": question,
                "response": response_text,
                "metadata": metadata,
                "error": final_state.get("error"),
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            error_metadata: PipelineMetadata = {
                "intent": "unknown",
                "intent_confidence": 0.0,
                "entities": {},
                "resolved_entities": [],
                "cypher_query": "",
                "cypher_parameters": {},
                "result_count": 0,
                "execution_path": ["pipeline_error"],
                "query_plan": None,
                "error": str(e),
            }
            return {
                "success": False,
                "question": question,
                "response": "죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.",
                "metadata": error_metadata,
                "error": str(e),
            }

    async def run_with_streaming(
        self,
        question: str,
        session_id: str | None = None,
        user_context: UserContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """스트리밍 모드로 파이프라인 실행"""
        logger.info(f"Running pipeline (streaming) for: {question[:50]}...")

        thread_id = session_id or str(uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        initial_state: GraphRAGState = {
            "question": question,
            "session_id": session_id or "",
            "messages": [HumanMessage(content=question)],
            "execution_path": [],
        }
        if self._graph_schema:
            initial_state["schema"] = self._graph_schema
        if user_context is not None:
            initial_state["user_context"] = user_context

        try:
            async for event in self._graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    yield {
                        "node": node_name,
                        "output": node_output,
                    }

        except Exception as e:
            logger.error(f"Pipeline streaming failed: {e}")
            yield {
                "node": "error",
                "output": {"error": str(e)},
            }

    async def run_with_streaming_response(
        self,
        question: str,
        session_id: str | None = None,
        user_context: UserContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """스트리밍 응답 파이프라인 (SSE 이벤트 형식)"""
        logger.info(f"Running streaming pipeline for: {question[:50]}...")

        thread_id = session_id or str(uuid4())
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        initial_state: GraphRAGState = {
            "question": question,
            "session_id": session_id or "",
            "messages": [HumanMessage(content=question)],
            "execution_path": [],
        }
        if self._graph_schema:
            initial_state["schema"] = self._graph_schema
        if user_context is not None:
            initial_state["user_context"] = user_context

        try:
            final_state: dict[str, Any] = dict(initial_state)
            accumulated_path: list[str] = []
            step_count = 0

            async for event in self._graph.astream(initial_state, config=config):
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        new_path_entries = node_output.get("execution_path", [])
                        if new_path_entries:
                            accumulated_path.extend(new_path_entries)
                            node_output = {
                                **node_output,
                                "execution_path": accumulated_path.copy(),
                            }
                        final_state.update(node_output)

                    step_count += 1
                    yield {
                        "type": "step",
                        "data": {
                            "node_name": node_name,
                            "description": _build_step_description(
                                node_name,
                                final_state,
                                node_output if isinstance(node_output, dict) else {},
                            ),
                            "step_number": step_count,
                        },
                    }

                    if node_name in ("response_generator", "clarification_handler"):
                        response = final_state.get("response", "")
                        if response:
                            yield {
                                "type": "metadata",
                                "data": self._build_metadata(final_state),
                            }
                            yield {"type": "chunk", "text": response}
                            yield {
                                "type": "done",
                                "full_response": response,
                                "success": True,
                            }
                            return

            existing_response = final_state.get("response", "")
            if existing_response:
                yield {"type": "metadata", "data": self._build_metadata(final_state)}
                yield {"type": "chunk", "text": existing_response}
                yield {
                    "type": "done",
                    "full_response": existing_response,
                    "success": True,
                }
                return

            metadata = self._build_metadata(final_state)
            yield {"type": "metadata", "data": metadata}

            step_count += 1
            yield {
                "type": "step",
                "data": {
                    "node_name": "response_generator",
                    "description": "응답 생성",
                    "step_number": step_count,
                },
            }

            cypher_query = final_state.get("cypher_query", "")
            graph_results = final_state.get("graph_results", [])
            error = final_state.get("error")

            if error or not cypher_query:
                fallback_response = (
                    f"질문을 처리할 수 없습니다: {error}"
                    if error
                    else "검색 조건에 맞는 결과를 찾지 못했습니다."
                )
                yield {"type": "chunk", "text": fallback_response}
                yield {
                    "type": "done",
                    "full_response": fallback_response,
                    "success": not bool(error),
                }
                return

            messages = final_state.get("messages", [])
            chat_history = format_chat_history(messages)

            full_response = ""
            async for chunk in self._llm.generate_response_stream(
                question=question,
                query_results=graph_results,
                cypher_query=cypher_query,
                chat_history=chat_history,
            ):
                full_response += chunk
                yield {"type": "chunk", "text": chunk}

            yield {
                "type": "done",
                "full_response": full_response,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Streaming pipeline failed: {e}")
            yield {
                "type": "error",
                "message": str(e),
            }

    def _build_metadata(self, state: dict[str, Any]) -> dict[str, Any]:
        """스트리밍용 메타데이터 구성"""
        return self._metadata_builder.build_metadata(state)
