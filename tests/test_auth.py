"""
Auth layer tests

Covers: JWTHandler, PasswordHandler, check_permission, UserContext
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import jwt
import pytest

from src.auth.jwt_handler import JWTHandler
from src.auth.models import DEFAULT_ROLE_PERMISSIONS, UserContext, permissions_for_roles
from src.auth.password import PasswordHandler
from src.auth.permissions import check_permission
from src.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jwt_settings() -> MagicMock:
    """Minimal Settings mock for JWTHandler."""
    settings = MagicMock(spec=Settings)
    # 32+ bytes to avoid PyJWT InsecureKeyLengthWarning in test output
    settings.jwt_secret_key = "test-secret-key-that-is-long-enough-for-hs256"
    settings.jwt_access_token_expire_minutes = 60
    settings.jwt_refresh_token_expire_days = 7
    return settings


@pytest.fixture
def jwt_handler(jwt_settings: MagicMock) -> JWTHandler:
    return JWTHandler(jwt_settings)


# ---------------------------------------------------------------------------
# JWTHandler
# ---------------------------------------------------------------------------


class TestJWTHandlerAccessToken:
    def test_create_access_token_returns_decodable_payload(self, jwt_handler: JWTHandler) -> None:
        payload = {"sub": "user-1", "roles": ["viewer"]}
        token = jwt_handler.create_access_token(payload)
        decoded = jwt_handler.decode_token(token)

        assert decoded["sub"] == "user-1"
        assert decoded["roles"] == ["viewer"]
        assert decoded["type"] == "access"

    def test_create_access_token_sets_expiry(self, jwt_handler: JWTHandler) -> None:
        token = jwt_handler.create_access_token({"sub": "user-1"})
        decoded = jwt_handler.decode_token(token)

        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        now = datetime.now(UTC)
        # expiry should be ~60 min in the future (allow ±5 s drift)
        assert timedelta(minutes=55) < (exp - now) < timedelta(minutes=65)

    def test_create_access_token_does_not_mutate_original_payload(
        self, jwt_handler: JWTHandler
    ) -> None:
        payload: dict = {"sub": "user-1"}
        jwt_handler.create_access_token(payload)
        assert "exp" not in payload
        assert "type" not in payload


class TestJWTHandlerRefreshToken:
    def test_create_refresh_token_returns_decodable_payload(self, jwt_handler: JWTHandler) -> None:
        token = jwt_handler.create_refresh_token("user-1")
        decoded = jwt_handler.decode_token(token)

        assert decoded["sub"] == "user-1"
        assert decoded["type"] == "refresh"

    def test_create_refresh_token_sets_expiry(self, jwt_handler: JWTHandler) -> None:
        token = jwt_handler.create_refresh_token("user-1")
        decoded = jwt_handler.decode_token(token)

        exp = datetime.fromtimestamp(decoded["exp"], tz=UTC)
        now = datetime.now(UTC)
        assert timedelta(days=6, hours=23) < (exp - now) < timedelta(days=7, hours=1)


class TestJWTHandlerDecodeToken:
    def test_decode_token_raises_on_expired_token(self, jwt_handler: JWTHandler) -> None:
        expired_payload = {
            "sub": "user-1",
            "exp": datetime.now(UTC) - timedelta(seconds=1),
            "type": "access",
        }
        secret = "test-secret-key-that-is-long-enough-for-hs256"
        token = jwt.encode(expired_payload, secret, algorithm="HS256")

        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_handler.decode_token(token)

    def test_decode_token_raises_on_wrong_signature(self, jwt_handler: JWTHandler) -> None:
        token = jwt.encode({"sub": "user-1"}, "wrong-secret", algorithm="HS256")

        with pytest.raises(jwt.InvalidTokenError):
            jwt_handler.decode_token(token)

    def test_decode_token_raises_on_malformed_token(self, jwt_handler: JWTHandler) -> None:
        with pytest.raises(jwt.InvalidTokenError):
            jwt_handler.decode_token("not.a.valid.jwt")


# ---------------------------------------------------------------------------
# PasswordHandler
# ---------------------------------------------------------------------------


class TestPasswordHandler:
    def test_hash_password_returns_bcrypt_hash(self) -> None:
        hashed = PasswordHandler.hash_password("secret")
        assert hashed.startswith("$2b$")

    def test_hash_password_is_not_plaintext(self) -> None:
        hashed = PasswordHandler.hash_password("secret")
        assert hashed != "secret"

    def test_hash_password_produces_unique_hashes(self) -> None:
        # bcrypt uses a random salt each call
        h1 = PasswordHandler.hash_password("secret")
        h2 = PasswordHandler.hash_password("secret")
        assert h1 != h2

    def test_verify_password_correct_password(self) -> None:
        hashed = PasswordHandler.hash_password("correct-password")
        assert PasswordHandler.verify_password("correct-password", hashed) is True

    def test_verify_password_wrong_password(self) -> None:
        hashed = PasswordHandler.hash_password("correct-password")
        assert PasswordHandler.verify_password("wrong-password", hashed) is False

    def test_verify_password_empty_string_does_not_match_non_empty(self) -> None:
        hashed = PasswordHandler.hash_password("secret")
        assert PasswordHandler.verify_password("", hashed) is False


# ---------------------------------------------------------------------------
# check_permission
# ---------------------------------------------------------------------------


class TestCheckPermission:
    def test_wildcard_granted_matches_any_permission(self) -> None:
        assert check_permission("*", "admin:users/write") is True
        assert check_permission("*", "node:Person/read") is True

    def test_exact_match(self) -> None:
        assert check_permission("node:Person/read", "node:Person/read") is True

    def test_exact_match_no_match(self) -> None:
        assert check_permission("node:Person/read", "node:Person/write") is False

    def test_resource_wildcard_matches_any_resource(self) -> None:
        assert check_permission("node:*/read", "node:Person/read") is True
        assert check_permission("node:*/read", "node:Organization/read") is True

    def test_resource_wildcard_does_not_match_different_action(self) -> None:
        assert check_permission("node:*/read", "node:Person/write") is False

    def test_action_wildcard_matches_any_action(self) -> None:
        assert check_permission("admin:*", "admin:users/read") is True
        assert check_permission("admin:*", "admin:settings/write") is True

    def test_action_wildcard_does_not_match_different_prefix(self) -> None:
        assert check_permission("admin:*", "node:Person/read") is False

    def test_case_sensitive_matching(self) -> None:
        # fnmatchcase must be used — uppercase should not match lowercase pattern
        assert check_permission("node:*/read", "Node:Person/read") is False
        assert check_permission("node:*/Read", "node:Person/read") is False

    def test_query_wildcard_pattern(self) -> None:
        assert check_permission("query:*/search", "query:entity_search/search") is True
        assert check_permission("query:*/search", "query:entity_search/write") is False


# ---------------------------------------------------------------------------
# UserContext
# ---------------------------------------------------------------------------


class TestUserContextHasPermission:
    def test_admin_user_has_all_permissions(self) -> None:
        user = UserContext(
            user_id="1",
            username="admin",
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
        )
        assert user.has_permission("node:Person", "read") is True
        assert user.has_permission("admin:users", "delete") is True

    def test_viewer_has_read_permission(self) -> None:
        user = UserContext.from_demo_role("viewer")
        assert user.has_permission("node:Person", "read") is True
        assert user.has_permission("query:entity_search", "search") is True

    def test_viewer_does_not_have_write_permission(self) -> None:
        user = UserContext.from_demo_role("viewer")
        assert user.has_permission("graph:edit", "write") is False

    def test_non_admin_is_admin_flag_false_enforces_permission_check(self) -> None:
        user = UserContext(
            user_id="1",
            username="viewer",
            roles=["viewer"],
            permissions=["node:*/read"],
            is_admin=False,
        )
        assert user.has_permission("node:Person", "read") is True
        assert user.has_permission("node:Person", "write") is False


class TestUserContextFromDemoRole:
    def test_from_demo_role_sets_correct_ids(self) -> None:
        user = UserContext.from_demo_role("viewer")
        assert user.user_id == "demo_viewer"
        assert user.username == "demo_viewer"

    def test_from_demo_role_admin_sets_is_admin_true(self) -> None:
        user = UserContext.from_demo_role("admin")
        assert user.is_admin is True
        assert user.permissions == ["*"]

    def test_from_demo_role_non_admin_sets_is_admin_false(self) -> None:
        user = UserContext.from_demo_role("viewer")
        assert user.is_admin is False

    def test_from_demo_role_unknown_role_gets_empty_permissions(self) -> None:
        user = UserContext.from_demo_role("unknown_role")
        assert user.permissions == []
        assert user.is_admin is False


class TestUserContextAnonymousAdmin:
    def test_anonymous_admin_is_admin(self) -> None:
        user = UserContext.anonymous_admin()
        assert user.is_admin is True

    def test_anonymous_admin_has_wildcard_permission(self) -> None:
        user = UserContext.anonymous_admin()
        assert user.permissions == ["*"]

    def test_anonymous_admin_identity(self) -> None:
        user = UserContext.anonymous_admin()
        assert user.user_id == "anonymous"
        assert user.username == "anonymous_admin"
        assert "admin" in user.roles


# ---------------------------------------------------------------------------
# permissions_for_roles
# ---------------------------------------------------------------------------


class TestPermissionsForRoles:
    def test_single_role_returns_correct_permissions(self) -> None:
        perms = permissions_for_roles(["viewer"])
        assert set(perms) == set(DEFAULT_ROLE_PERMISSIONS["viewer"])

    def test_admin_role_returns_wildcard(self) -> None:
        perms = permissions_for_roles(["admin"])
        assert perms == ["*"]

    def test_multiple_roles_merges_without_duplicates(self) -> None:
        # manager and editor share the same permissions
        perms = permissions_for_roles(["manager", "editor"])
        assert len(perms) == len(set(perms)), "Duplicates found in merged permissions"

    def test_unknown_role_returns_empty_list(self) -> None:
        assert permissions_for_roles(["nonexistent"]) == []

    def test_empty_roles_returns_empty_list(self) -> None:
        assert permissions_for_roles([]) == []

    def test_order_is_preserved_first_role_wins(self) -> None:
        # The first role's permissions should appear before the second role's unique ones
        perms = permissions_for_roles(["viewer", "manager"])
        viewer_perms = DEFAULT_ROLE_PERMISSIONS["viewer"]
        for p in viewer_perms:
            assert p in perms
