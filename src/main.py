"""
Graph RAG Core API

FastAPI 애플리케이션 진입점
도메인 독립적 Graph RAG 백엔드 템플릿
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import graph_edit_router, query_router
from src.api.services.explainability import ExplainabilityService
from src.auth.jwt_handler import JWTHandler
from src.auth.password import PasswordHandler
from src.config import get_settings
from src.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    EntityNotFoundError,
    GraphRAGError,
)
from src.graph import GraphRAGPipeline
from src.graph.checkpointer import create_checkpointer
from src.infrastructure.neo4j_client import Neo4jClient
from src.repositories import LLMRepository, Neo4jRepository
from src.repositories.user_repository import UserRepository
from src.services.auth_service import AuthService
from src.services.graph_edit_service import GraphEditService

# 로깅 설정
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    애플리케이션 라이프사이클 관리

    시작 시: Neo4j 연결, Pipeline 초기화
    종료 시: 리소스 정리
    """
    logger.info("Starting Graph RAG Core API...")

    # Neo4j 클라이언트 초기화
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        connection_timeout=settings.neo4j_connection_timeout,
    )
    await neo4j_client.connect()
    logger.info("Neo4j client connected")

    # Repository 초기화
    neo4j_repo = Neo4jRepository(neo4j_client)
    llm_repo = LLMRepository(settings)

    # 스키마 사전 로드 (파이프라인에 주입)
    try:
        graph_schema = await neo4j_repo.get_schema()
        logger.info(
            f"Schema loaded: "
            f"{len(graph_schema.get('node_labels', []))} labels, "
            f"{len(graph_schema.get('relationship_types', []))} relationships"
        )
    except Exception as e:
        logger.error(f"Failed to load graph schema: {e}")
        await neo4j_client.close()
        logger.info("Neo4j connection closed due to schema loading failure")
        raise RuntimeError(
            "Schema loading is required for pipeline initialization"
        ) from e

    # Checkpointer 초기화
    checkpointer = await create_checkpointer(settings.checkpointer_db_path)
    logger.info(f"Checkpointer initialized: {type(checkpointer).__name__}")

    # Pipeline 초기화 (스키마 + checkpointer 주입)
    pipeline = GraphRAGPipeline(
        settings=settings,
        neo4j_repository=neo4j_repo,
        llm_repository=llm_repo,
        neo4j_client=neo4j_client,
        graph_schema=graph_schema,
        checkpointer=checkpointer,
    )
    logger.info("Pipeline initialized with pre-loaded schema")

    # ExplainabilityService 초기화 (stateless)
    explainability_service = ExplainabilityService()
    logger.info("ExplainabilityService initialized")

    # GraphEditService 초기화 (동적 스키마)
    graph_edit_service = GraphEditService(neo4j_repo, neo4j_client)
    logger.info("GraphEditService initialized")

    # AuthService 초기화
    user_repository = UserRepository(neo4j_client)
    jwt_handler = JWTHandler(settings)
    password_handler = PasswordHandler()
    auth_service = AuthService(
        user_repository=user_repository,
        jwt_handler=jwt_handler,
        password_handler=password_handler,
        settings=settings,
    )
    logger.info(f"AuthService initialized (auth_enabled={settings.auth_enabled})")

    # app.state에 저장
    app.state.neo4j_client = neo4j_client
    app.state.neo4j_repo = neo4j_repo
    app.state.llm_repo = llm_repo
    app.state.pipeline = pipeline
    app.state.explainability_service = explainability_service
    app.state.graph_edit_service = graph_edit_service
    app.state.auth_service = auth_service

    yield

    # 종료 시 리소스 정리
    logger.info("Shutting down Graph RAG Core API...")

    if hasattr(app.state, "llm_repo") and app.state.llm_repo:
        await app.state.llm_repo.close()
        logger.info("LLM client closed")

    if hasattr(app.state, "neo4j_client") and app.state.neo4j_client:
        await app.state.neo4j_client.close()
        logger.info("Neo4j connection closed")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Neo4j 그래프 데이터베이스와 Azure OpenAI를 활용한 도메인 독립적 RAG 시스템",
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID", "X-Demo-Role"],
)


# ============================================
# 글로벌 예외 핸들러
# ============================================


@app.exception_handler(AuthenticationError)
async def authentication_error_handler(
    request: Request, exc: AuthenticationError
) -> JSONResponse:
    """인증 실패 시 401 응답"""
    return JSONResponse(
        status_code=401,
        content={"detail": {"message": exc.message, "code": exc.code}},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(AuthorizationError)
async def authorization_error_handler(
    request: Request, exc: AuthorizationError
) -> JSONResponse:
    """인가 실패 시 403 응답"""
    return JSONResponse(
        status_code=403,
        content={"detail": {"message": exc.message, "code": exc.code}},
    )


@app.exception_handler(EntityNotFoundError)
async def entity_not_found_handler(
    request: Request, exc: EntityNotFoundError
) -> JSONResponse:
    """엔티티를 찾을 수 없을 때 404 응답"""
    return JSONResponse(
        status_code=404,
        content={"detail": {"message": exc.message, "code": exc.code}},
    )


@app.exception_handler(GraphRAGError)
async def graphrag_error_handler(request: Request, exc: GraphRAGError) -> JSONResponse:
    """기타 도메인 예외 시 500 응답"""
    settings = get_settings()
    if settings.is_production:
        logger.error(f"GraphRAGError: {exc.code}")
    else:
        logger.error(f"GraphRAGError: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": {"message": exc.message, "code": exc.code}},
    )


# 라우터 등록
app.include_router(query_router)
app.include_router(graph_edit_router)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
    )
