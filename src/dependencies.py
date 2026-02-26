"""
FastAPI 의존성 주입 모듈

FastAPI의 Depends 패턴을 활용한 의존성 주입을 관리합니다.

의존성 흐름:
    Settings -> Infrastructure Clients -> Repositories -> Pipeline -> Services
"""

import logging
from typing import TYPE_CHECKING

from fastapi import Request

from src.auth.models import UserContext
from src.config import get_settings
from src.domain.exceptions import AuthenticationError
from src.graph.pipeline import GraphRAGPipeline
from src.infrastructure.neo4j_client import Neo4jClient
from src.repositories.llm_repository import LLMRepository
from src.repositories.neo4j_repository import Neo4jRepository
from src.services.graph_edit_service import GraphEditService

if TYPE_CHECKING:
    from src.api.services.explainability import ExplainabilityService
    from src.services.auth_service import AuthService

logger = logging.getLogger(__name__)


# ============================================
# Infrastructure Layer 의존성
# ============================================


def get_neo4j_client(request: Request) -> Neo4jClient:
    """Neo4j 클라이언트 의존성 주입"""
    return request.app.state.neo4j_client


# ============================================
# Repository Layer 의존성
# ============================================


def get_neo4j_repository(request: Request) -> Neo4jRepository:
    """Neo4j Repository 의존성 주입"""
    return request.app.state.neo4j_repo


def get_llm_repository(request: Request) -> LLMRepository:
    """LLM Repository 의존성 주입"""
    return request.app.state.llm_repo


# ============================================
# Graph Pipeline 의존성
# ============================================


def get_graph_pipeline(request: Request) -> GraphRAGPipeline:
    """GraphRAG Pipeline 의존성 주입"""
    return request.app.state.pipeline


# ============================================
# Service 의존성
# ============================================


def get_graph_edit_service(request: Request) -> GraphEditService:
    """GraphEditService 의존성 주입"""
    return request.app.state.graph_edit_service


def get_explainability_service(request: Request) -> "ExplainabilityService":
    """ExplainabilityService 의존성 주입"""
    return request.app.state.explainability_service


def get_auth_service(request: Request) -> "AuthService":
    """AuthService 의존성 주입"""
    return request.app.state.auth_service


# ============================================
# 인증 관련 의존성
# ============================================

ALLOWED_DEMO_ROLES = {"admin", "manager", "editor", "viewer"}


async def get_current_user(request: Request) -> UserContext:
    """
    현재 요청의 인증된 사용자 컨텍스트를 반환

    AUTH_ENABLED=false (데모/개발 모드):
        X-Demo-Role 헤더 → 데모 UserContext (허용된 역할만)
        헤더 없음 → anonymous_admin

    AUTH_ENABLED=true (프로덕션):
        X-Demo-Role 무시, Bearer 토큰 필수 → AuthService.get_current_user()

    Raises:
        AuthenticationError: 토큰 없음, 만료, 유효하지 않음
    """
    settings = get_settings()

    if not settings.auth_enabled:
        demo_role = request.headers.get("X-Demo-Role")
        if demo_role and demo_role in ALLOWED_DEMO_ROLES:
            return UserContext.from_demo_role(demo_role)
        return UserContext.anonymous_admin()

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Missing or invalid Authorization header")

    token = auth_header[len("Bearer ") :]
    if not token:
        raise AuthenticationError("Empty token")

    auth_service: AuthService = request.app.state.auth_service
    return await auth_service.get_current_user(token)
