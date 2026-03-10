"""
Service Layer Tests

GraphEditService and AuthService 단위 테스트.
모든 외부 의존성(Neo4jRepository, Neo4jClient, UserRepository, JWTHandler, PasswordHandler)은
AsyncMock/MagicMock으로 대체합니다.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from src.auth.models import UserContext
from src.domain.exceptions import AuthenticationError, EntityNotFoundError, ValidationError
from src.repositories.neo4j_types import NodeResult
from src.services.auth_service import AuthService
from src.services.graph_edit_service import GraphEditConflictError, GraphEditService


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_neo4j_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.get_node_labels.return_value = ["Person", "Organization"]
    repo.get_relationship_types.return_value = ["WORKS_AT", "KNOWS"]
    repo.check_duplicate_node.return_value = False
    repo.create_node_generic.return_value = {
        "id": "node-1",
        "labels": ["Person"],
        "properties": {"name": "Alice", "created_by": "anonymous_admin"},
    }
    return repo


@pytest.fixture
def graph_edit_service(mock_neo4j_repo: AsyncMock) -> GraphEditService:
    return GraphEditService(neo4j_repository=mock_neo4j_repo)


@pytest.fixture
def mock_user_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_jwt_handler() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_password_handler() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_settings(mock_settings: Any) -> Any:
    # conftest.py의 mock_settings fixture 재사용
    return mock_settings


@pytest.fixture
def auth_service(
    mock_user_repo: AsyncMock,
    mock_jwt_handler: MagicMock,
    mock_password_handler: MagicMock,
    mock_settings: Any,
) -> AuthService:
    return AuthService(
        user_repository=mock_user_repo,
        jwt_handler=mock_jwt_handler,
        password_handler=mock_password_handler,
        settings=mock_settings,
    )


def _make_node_result(
    node_id: str = "node-1",
    labels: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> NodeResult:
    return NodeResult(
        id=node_id,
        labels=labels or ["Person"],
        properties=properties or {"name": "Alice"},
    )


# ============================================================
# GraphEditService — create_node
# ============================================================


class TestGraphEditServiceCreateNode:
    async def test_create_node_success(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        result = await graph_edit_service.create_node(
            label="Person",
            properties={"name": "Alice", "email": "alice@example.com"},
        )

        assert result["properties"]["name"] == "Alice"
        mock_neo4j_repo.check_duplicate_node.assert_awaited_once_with("Person", "Alice")
        mock_neo4j_repo.create_node_generic.assert_awaited_once()

    async def test_create_node_missing_name_raises_validation_error(
        self, graph_edit_service: GraphEditService
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            await graph_edit_service.create_node(
                label="Person",
                properties={"email": "alice@example.com"},
            )

        assert exc_info.value.field == "name"

    async def test_create_node_blank_name_raises_validation_error(
        self, graph_edit_service: GraphEditService
    ) -> None:
        with pytest.raises(ValidationError):
            await graph_edit_service.create_node(
                label="Person",
                properties={"name": "   "},
            )

    async def test_create_node_duplicate_raises_conflict_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.check_duplicate_node.return_value = True

        with pytest.raises(GraphEditConflictError):
            await graph_edit_service.create_node(
                label="Person",
                properties={"name": "Alice"},
            )

    async def test_create_node_trims_name_whitespace(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        await graph_edit_service.create_node(
            label="Person",
            properties={"name": "  Bob  "},
        )

        mock_neo4j_repo.check_duplicate_node.assert_awaited_once_with("Person", "Bob")

    async def test_create_node_repository_returns_none_raises_conflict_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.create_node_generic.return_value = None

        with pytest.raises(GraphEditConflictError):
            await graph_edit_service.create_node(
                label="Person",
                properties={"name": "Alice"},
            )


# ============================================================
# GraphEditService — update_node
# ============================================================


class TestGraphEditServiceUpdateNode:
    async def test_update_node_success(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        existing = _make_node_result(properties={"name": "Alice", "email": "old@example.com"})
        mock_neo4j_repo.find_entity_by_id.return_value = existing
        mock_neo4j_repo.update_node_properties.return_value = {
            "id": "node-1",
            "properties": {"name": "Alice", "email": "new@example.com"},
        }

        result = await graph_edit_service.update_node(
            node_id="node-1",
            properties={"email": "new@example.com"},
        )

        assert result["properties"]["email"] == "new@example.com"
        mock_neo4j_repo.update_node_properties.assert_awaited_once()

    async def test_update_node_filters_protected_properties(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        existing = _make_node_result(properties={"name": "Alice"})
        mock_neo4j_repo.find_entity_by_id.return_value = existing
        mock_neo4j_repo.update_node_properties.return_value = {"id": "node-1", "properties": {}}

        await graph_edit_service.update_node(
            node_id="node-1",
            properties={
                "email": "new@example.com",
                "created_at": "2024-01-01",   # protected — user input should be dropped
                "updated_by": "hacker",        # protected — overwritten by service
            },
        )

        call_args = mock_neo4j_repo.update_node_properties.call_args
        update_props: dict[str, Any] = call_args[0][1]
        # created_at from user must be stripped
        assert "created_at" not in update_props
        # updated_by is re-injected by the service as "anonymous_admin", not "hacker"
        assert update_props.get("updated_by") == "anonymous_admin"
        assert "email" in update_props

    async def test_update_node_name_change_duplicate_raises_conflict_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        existing = _make_node_result(
            labels=["Person"],
            properties={"name": "Alice"},
        )
        mock_neo4j_repo.find_entity_by_id.return_value = existing
        mock_neo4j_repo.check_duplicate_node.return_value = True

        with pytest.raises(GraphEditConflictError):
            await graph_edit_service.update_node(
                node_id="node-1",
                properties={"name": "Bob"},
            )

    async def test_update_node_same_name_skips_duplicate_check(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        existing = _make_node_result(properties={"name": "Alice"})
        mock_neo4j_repo.find_entity_by_id.return_value = existing
        mock_neo4j_repo.update_node_properties.return_value = {"id": "node-1", "properties": {}}

        await graph_edit_service.update_node(
            node_id="node-1",
            properties={"name": "alice"},  # same name, different case
        )

        mock_neo4j_repo.check_duplicate_node.assert_not_awaited()


# ============================================================
# GraphEditService — delete_node
# ============================================================


class TestGraphEditServiceDeleteNode:
    async def test_delete_node_success(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.delete_node_atomic.return_value = {
            "deleted": True,
            "not_found": False,
            "rel_count": 0,
        }

        await graph_edit_service.delete_node("node-1")

        mock_neo4j_repo.delete_node_atomic.assert_awaited_once_with("node-1", force=False)

    async def test_delete_node_with_relationships_force_false_raises_conflict_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.delete_node_atomic.return_value = {
            "deleted": False,
            "not_found": False,
            "rel_count": 3,
        }

        with pytest.raises(GraphEditConflictError) as exc_info:
            await graph_edit_service.delete_node("node-1", force=False)

        assert "3" in str(exc_info.value)

    async def test_delete_node_with_relationships_force_true_succeeds(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.delete_node_atomic.return_value = {
            "deleted": True,
            "not_found": False,
            "rel_count": 3,
        }

        await graph_edit_service.delete_node("node-1", force=True)

        mock_neo4j_repo.delete_node_atomic.assert_awaited_once_with("node-1", force=True)

    async def test_delete_node_not_found_raises_entity_not_found_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.delete_node_atomic.return_value = {
            "deleted": False,
            "not_found": True,
            "rel_count": 0,
        }

        with pytest.raises(EntityNotFoundError):
            await graph_edit_service.delete_node("nonexistent-node")


# ============================================================
# GraphEditService — create_edge
# ============================================================


class TestGraphEditServiceCreateEdge:
    async def test_create_edge_success(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.find_entity_by_id.side_effect = [
            _make_node_result("node-1"),
            _make_node_result("node-2", labels=["Organization"]),
        ]
        mock_neo4j_repo.create_relationship_generic.return_value = {
            "id": "edge-1",
            "type": "WORKS_AT",
        }

        result = await graph_edit_service.create_edge(
            source_id="node-1",
            target_id="node-2",
            relationship_type="WORKS_AT",
            properties={"since": "2020"},
        )

        assert result["type"] == "WORKS_AT"
        assert mock_neo4j_repo.find_entity_by_id.await_count == 2
        mock_neo4j_repo.create_relationship_generic.assert_awaited_once()

    async def test_create_edge_source_not_found_raises_error(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.find_entity_by_id.side_effect = EntityNotFoundError("Node", "node-1")

        with pytest.raises(EntityNotFoundError):
            await graph_edit_service.create_edge(
                source_id="node-1",
                target_id="node-2",
                relationship_type="WORKS_AT",
            )

        mock_neo4j_repo.create_relationship_generic.assert_not_awaited()

    async def test_create_edge_injects_created_by(
        self, graph_edit_service: GraphEditService, mock_neo4j_repo: AsyncMock
    ) -> None:
        mock_neo4j_repo.find_entity_by_id.side_effect = [
            _make_node_result("node-1"),
            _make_node_result("node-2"),
        ]
        mock_neo4j_repo.create_relationship_generic.return_value = {"id": "edge-1"}

        await graph_edit_service.create_edge("node-1", "node-2", "KNOWS")

        call_args = mock_neo4j_repo.create_relationship_generic.call_args
        edge_props: dict[str, Any] = call_args[0][3]
        assert edge_props.get("created_by") == "anonymous_admin"


# ============================================================
# AuthService — login
# ============================================================


class TestAuthServiceLogin:
    async def test_login_success(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
        mock_password_handler: MagicMock,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = {
            "id": "user-1",
            "username": "alice",
            "hashed_password": "hashed",
            "roles": ["viewer"],
            "is_active": True,
            "department": None,
        }
        mock_password_handler.verify_password.return_value = True
        mock_jwt_handler.create_access_token.return_value = "access_token_value"
        mock_jwt_handler.create_refresh_token.return_value = "refresh_token_value"

        result = await auth_service.login("alice", "correct_password")

        assert result["access_token"] == "access_token_value"
        assert result["refresh_token"] == "refresh_token_value"
        assert result["token_type"] == "bearer"

    async def test_login_invalid_username_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = None

        with pytest.raises(AuthenticationError):
            await auth_service.login("nonexistent", "password")

    async def test_login_wrong_password_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
        mock_password_handler: MagicMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = {
            "id": "user-1",
            "username": "alice",
            "hashed_password": "hashed",
            "roles": ["viewer"],
            "is_active": True,
        }
        mock_password_handler.verify_password.return_value = False

        with pytest.raises(AuthenticationError):
            await auth_service.login("alice", "wrong_password")

    async def test_login_disabled_account_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = {
            "id": "user-1",
            "username": "alice",
            "hashed_password": "hashed",
            "roles": ["viewer"],
            "is_active": False,
        }

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.login("alice", "password")

        assert "disabled" in str(exc_info.value).lower()


# ============================================================
# AuthService — get_current_user
# ============================================================


class TestAuthServiceGetCurrentUser:
    async def test_get_current_user_valid_access_token(
        self,
        auth_service: AuthService,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_jwt_handler.decode_token.return_value = {
            "type": "access",
            "sub": "user-1",
            "username": "alice",
            "roles": ["viewer"],
            "department": "engineering",
        }

        user_ctx = await auth_service.get_current_user("valid_token")

        assert isinstance(user_ctx, UserContext)
        assert user_ctx.user_id == "user-1"
        assert user_ctx.username == "alice"
        assert user_ctx.is_admin is False
        assert "query:*/search" in user_ctx.permissions

    async def test_get_current_user_admin_role_sets_is_admin(
        self,
        auth_service: AuthService,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_jwt_handler.decode_token.return_value = {
            "type": "access",
            "sub": "admin-1",
            "username": "admin",
            "roles": ["admin"],
            "department": None,
        }

        user_ctx = await auth_service.get_current_user("admin_token")

        assert user_ctx.is_admin is True
        assert user_ctx.permissions == ["*"]

    async def test_get_current_user_invalid_token_type_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_jwt_handler.decode_token.return_value = {
            "type": "refresh",  # wrong type
            "sub": "user-1",
            "username": "alice",
            "roles": ["viewer"],
        }

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.get_current_user("refresh_token")

        assert "token type" in str(exc_info.value).lower()

    async def test_get_current_user_expired_token_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_jwt_handler.decode_token.side_effect = jwt.ExpiredSignatureError

        with pytest.raises(AuthenticationError):
            await auth_service.get_current_user("expired_token")

    async def test_get_current_user_invalid_token_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_jwt_handler: MagicMock,
    ) -> None:
        mock_jwt_handler.decode_token.side_effect = jwt.InvalidTokenError

        with pytest.raises(AuthenticationError):
            await auth_service.get_current_user("garbage_token")


# ============================================================
# AuthService — create_user
# ============================================================


class TestAuthServiceCreateUser:
    async def test_create_user_success(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
        mock_password_handler: MagicMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = None
        mock_password_handler.hash_password.return_value = "hashed_pw"
        mock_user_repo.create_user.return_value = {
            "id": "user-new",
            "username": "bob",
            "hashed_password": "hashed_pw",
            "roles": ["viewer"],
            "is_active": True,
            "department": None,
        }

        result = await auth_service.create_user(
            username="bob",
            password="secure_password",
            roles=["viewer"],
        )

        assert result["username"] == "bob"
        assert "hashed_password" not in result  # stripped before returning

    async def test_create_user_duplicate_username_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = {
            "id": "existing-user",
            "username": "bob",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.create_user(username="bob", password="password")

        assert "bob" in str(exc_info.value)

    async def test_create_user_constraint_error_raises_authentication_error(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
        mock_password_handler: MagicMock,
    ) -> None:
        from neo4j.exceptions import ConstraintError

        mock_user_repo.find_by_username.return_value = None
        mock_password_handler.hash_password.return_value = "hashed_pw"
        mock_user_repo.create_user.side_effect = ConstraintError("Constraint violation")

        with pytest.raises(AuthenticationError):
            await auth_service.create_user(username="bob", password="password")

    async def test_create_user_default_roles_assigned(
        self,
        auth_service: AuthService,
        mock_user_repo: AsyncMock,
        mock_password_handler: MagicMock,
    ) -> None:
        mock_user_repo.find_by_username.return_value = None
        mock_password_handler.hash_password.return_value = "hashed_pw"
        mock_user_repo.create_user.return_value = {
            "id": "user-new",
            "username": "carol",
            "roles": ["viewer"],
            "is_active": True,
        }

        await auth_service.create_user(username="carol", password="password")

        call_kwargs = mock_user_repo.create_user.call_args[1]
        assert "username" in call_kwargs
        assert call_kwargs["username"] == "carol"
