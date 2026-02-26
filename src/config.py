"""
애플리케이션 설정 모듈

Pydantic Settings를 활용한 환경변수 기반 설정 관리
- 타입 검증 자동화
- .env 파일 지원
- 환경별 설정 분리
"""

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 (src/config.py 기준으로 한 단계 상위)
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정

    환경변수 또는 .env 파일에서 값을 로드합니다.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ============================================
    # 애플리케이션 설정
    # ============================================
    app_name: str = Field(default="Graph RAG Core API", description="애플리케이션 이름")
    app_version: str = Field(default="0.1.0", description="애플리케이션 버전")
    debug: bool = Field(default=False, description="디버그 모드")
    environment: str = Field(default="development", description="실행 환경")
    cors_origins: list[str] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
        ],
        description="CORS 허용 오리진 목록",
    )

    # ============================================
    # Neo4j 설정
    # ============================================
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j 사용자명",
    )
    neo4j_password: str = Field(
        default="",
        description="Neo4j 비밀번호",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j 데이터베이스 이름",
    )
    neo4j_max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Neo4j 커넥션 풀 최대 크기",
    )
    neo4j_connection_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Neo4j 연결 타임아웃 (초)",
    )

    # ============================================
    # Azure OpenAI 설정
    # ============================================
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI 엔드포인트 URL",
    )
    azure_openai_api_key: str | None = Field(
        default=None,
        description="Azure OpenAI API 키 (Managed Identity 사용 시 불필요)",
    )
    azure_openai_api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI API 버전",
    )

    # 배포 이름
    light_model_deployment: str = Field(
        default="gpt-4o-mini",
        description="가벼운 작업용 모델 배포명",
    )
    heavy_model_deployment: str = Field(
        default="gpt-4o",
        description="복잡한 작업용 모델 배포명",
    )

    # LLM 공통 파라미터
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM 응답 온도",
    )
    llm_max_tokens: int = Field(
        default=2000,
        ge=1,
        le=16000,
        description="LLM 최대 토큰 수",
    )

    # ============================================
    # Azure OpenAI Embedding 설정
    # ============================================
    embedding_model_deployment: str = Field(
        default="text-embedding-3-small",
        description="Azure OpenAI embedding 모델 배포명",
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=256,
        le=3072,
        description="Embedding 벡터 차원",
    )

    # ============================================
    # Latency Optimization 설정
    # ============================================
    cypher_light_model_enabled: bool = Field(
        default=True,
        description="단순 쿼리에 LIGHT 모델 사용 여부",
    )

    # ============================================
    # Vector Search 설정
    # ============================================
    vector_similarity_threshold: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Vector Search 유사도 임계값",
    )
    query_cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="질문-Cypher 캐시 TTL (시간)",
    )
    vector_search_enabled: bool = Field(
        default=True,
        description="Vector Search 기능 활성화 여부",
    )

    # ============================================
    # Graph Edit 설정
    # ============================================
    graph_edit_enabled: bool = Field(
        default=True,
        description="그래프 편집 API 활성화 여부",
    )

    # ============================================
    # Authentication & Authorization 설정
    # ============================================
    auth_enabled: bool = Field(
        default=False,
        description="인증/인가 활성화 여부 (False면 AnonymousAdmin으로 전체 권한 부여)",
    )
    jwt_secret_key: str = Field(
        default="dev-insecure-secret-key-change-in-production",
        description="JWT 서명 키 (프로덕션에서는 반드시 변경)",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="JWT 액세스 토큰 만료 시간 (분)",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="JWT 리프레시 토큰 만료 시간 (일)",
    )
    graph_edit_require_auth: bool = Field(
        default=True,
        description="그래프 편집 시 인증 필수 여부",
    )

    # ============================================
    # Checkpointer 설정
    # ============================================
    checkpointer_db_path: str = Field(
        default="checkpoints.db",
        description="SQLite checkpointer DB 경로 (':memory:'면 MemorySaver 사용)",
    )

    # ============================================
    # 로깅 설정
    # ============================================
    log_level: str = Field(
        default="INFO",
        description="로깅 레벨",
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="로그 포맷",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """로그 레벨 유효성 검사"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper_v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """환경 유효성 검사"""
        valid_envs = {"development", "staging", "production", "test"}
        lower_v = v.lower()
        if lower_v not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")
        return lower_v

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(cls, v: list[str]) -> list[str]:
        """CORS 오리진 목록 검증"""
        if "*" in v and len(v) > 1:
            raise ValueError(
                "CORS origins cannot mix wildcard '*' with specific origins. "
                "Use either '*' alone or specific origin URLs."
            )
        return v

    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == "development"

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """프로덕션 환경 필수 설정 검증"""
        if self.environment == "production":
            missing_fields = []

            if not self.neo4j_password:
                missing_fields.append("neo4j_password")

            if not self.azure_openai_endpoint:
                missing_fields.append("azure_openai_endpoint")

            if not self.azure_openai_api_key:
                logger.warning(
                    "azure_openai_api_key is not set. "
                    "Ensure Managed Identity is configured for Azure OpenAI access."
                )

            if not self.auth_enabled:
                logger.warning(
                    "AUTH_ENABLED is false in production. "
                    "Ensure authentication is handled externally (VPN, API Gateway, etc.)."
                )

            if self.auth_enabled and self.jwt_secret_key.startswith("dev-insecure"):
                raise ValueError(
                    "jwt_secret_key is using default insecure value in production! "
                    "Set a strong secret key via JWT_SECRET_KEY environment variable."
                )

            if missing_fields:
                raise ValueError(
                    f"Production environment requires these settings: {', '.join(missing_fields)}"
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """설정 싱글톤 인스턴스 반환"""
    return Settings()
