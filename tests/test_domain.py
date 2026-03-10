"""
Domain Layer Tests

validators.py 및 exceptions.py에 대한 단위 테스트
"""

import pytest

from src.domain.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ConflictError,
    CypherGenerationError,
    DatabaseConnectionError,
    DatabaseError,
    EmptyResultError,
    EntityNotFoundError,
    EntityResolutionError,
    GraphRAGError,
    IntentClassificationError,
    InvalidQueryError,
    InvalidStateError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    PipelineError,
    QueryExecutionError,
    ValidationError,
)
from src.domain.validators import validate_cypher_identifier, validate_read_only_cypher


# ============================================================
# validate_read_only_cypher
# ============================================================


class TestValidateReadOnlyCypher:
    def test_valid_match_query_passes(self) -> None:
        query = "MATCH (n:Person) RETURN n.name"
        assert validate_read_only_cypher(query) == query

    def test_valid_match_with_where_passes(self) -> None:
        query = "MATCH (n:Movie) WHERE n.year > 2000 RETURN n"
        assert validate_read_only_cypher(query) == query

    def test_valid_with_unwind_passes(self) -> None:
        query = "MATCH (n) WITH n UNWIND [1,2] AS x RETURN x"
        assert validate_read_only_cypher(query) == query

    def test_create_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("CREATE (n:Person {name: 'Alice'})")

    def test_delete_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("MATCH (n) DELETE n")

    def test_detach_delete_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("MATCH (n) DETACH DELETE n")

    def test_set_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("MATCH (n) SET n.name = 'Bob'")

    def test_remove_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("MATCH (n) REMOVE n.age")

    def test_merge_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("MERGE (n:Person {name: 'Alice'})")

    def test_drop_keyword_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("DROP INDEX my_index")

    def test_call_subquery_no_space_raises(self) -> None:
        # Regex matches CALL{ (no space between CALL and {)
        with pytest.raises(ValueError):
            validate_read_only_cypher("CALL{MATCH (n) RETURN n}")

    def test_call_subquery_with_space_not_matched(self) -> None:
        # CALL { ... } with a space before { is not caught by the current regex —
        # document the actual behavior so this test acts as a spec.
        query = "CALL { MATCH (n) RETURN n }"
        result = validate_read_only_cypher(query)
        assert result == query

    def test_case_insensitive_lowercase_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("match (n) delete n")

    def test_case_insensitive_mixed_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_read_only_cypher("Match (n) Create (m:Node)")

    def test_write_keyword_as_substring_passes(self) -> None:
        # "creates" is not the keyword CREATE — word boundary must prevent match
        query = "MATCH (n) WHERE n.description = 'creates value' RETURN n"
        assert validate_read_only_cypher(query) == query

    def test_returns_original_query_unchanged(self) -> None:
        query = "MATCH (a)-[:KNOWS]->(b) RETURN a, b LIMIT 10"
        result = validate_read_only_cypher(query)
        assert result is query


# ============================================================
# validate_cypher_identifier
# ============================================================


class TestValidateCypherIdentifier:
    def test_ascii_identifier_passes(self) -> None:
        assert validate_cypher_identifier("Person") == "Person"

    def test_identifier_with_underscore_passes(self) -> None:
        assert validate_cypher_identifier("my_label") == "my_label"

    def test_identifier_with_digits_passes(self) -> None:
        assert validate_cypher_identifier("label123") == "label123"

    def test_korean_identifier_passes(self) -> None:
        assert validate_cypher_identifier("사람") == "사람"

    def test_mixed_ascii_korean_passes(self) -> None:
        assert validate_cypher_identifier("person사람") == "person사람"

    def test_starts_with_digit_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("1invalid")

    def test_starts_with_underscore_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("_hidden")

    def test_special_char_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("label-name")

    def test_space_in_identifier_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("my label")

    def test_injection_attempt_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("Person`) MATCH (n) DETACH DELETE n //")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            validate_cypher_identifier("")

    def test_field_name_appears_in_error_message(self) -> None:
        with pytest.raises(ValueError, match="node_label"):
            validate_cypher_identifier("1bad", field_name="node_label")

    def test_returns_original_value_unchanged(self) -> None:
        name = "Movie"
        result = validate_cypher_identifier(name)
        assert result is name


# ============================================================
# Exception hierarchy
# ============================================================


class TestGraphRAGErrorBase:
    def test_message_and_code_attributes(self) -> None:
        err = GraphRAGError("something went wrong", code="MY_CODE")
        assert err.message == "something went wrong"
        assert err.code == "MY_CODE"

    def test_default_code(self) -> None:
        err = GraphRAGError("oops")
        assert err.code == "UNKNOWN_ERROR"

    def test_is_exception(self) -> None:
        assert isinstance(GraphRAGError("x"), Exception)

    def test_str_returns_message(self) -> None:
        err = GraphRAGError("hello")
        assert str(err) == "hello"


class TestDatabaseExceptions:
    def test_database_error_is_graph_rag_error(self) -> None:
        err = DatabaseError("db fail")
        assert isinstance(err, GraphRAGError)
        assert err.code == "DATABASE_ERROR"

    def test_database_connection_error_code(self) -> None:
        err = DatabaseConnectionError("cannot connect")
        assert err.code == "DATABASE_CONNECTION_ERROR"
        assert isinstance(err, DatabaseError)

    def test_query_execution_error_code(self) -> None:
        err = QueryExecutionError("bad query", query="MATCH (n) RTRN n")
        assert err.code == "QUERY_EXECUTION_ERROR"
        assert err._query == "MATCH (n) RTRN n"

    def test_entity_not_found_error_attributes(self) -> None:
        err = EntityNotFoundError("Person", "abc-123")
        assert err.entity_type == "Person"
        assert err.entity_id == "abc-123"
        assert err.code == "ENTITY_NOT_FOUND"
        assert "abc-123" in err.message


class TestLLMExceptions:
    def test_llm_connection_error_code(self) -> None:
        err = LLMConnectionError("timeout")
        assert err.code == "LLM_CONNECTION_ERROR"
        assert isinstance(err, GraphRAGError)

    def test_llm_rate_limit_error_retry_after(self) -> None:
        err = LLMRateLimitError("rate limited", retry_after=30)
        assert err.code == "LLM_RATE_LIMIT"
        assert err.retry_after == 30

    def test_llm_rate_limit_default_retry_after(self) -> None:
        err = LLMRateLimitError("rate limited")
        assert err.retry_after == 0

    def test_llm_response_error_code(self) -> None:
        err = LLMResponseError("bad response")
        assert err.code == "LLM_RESPONSE_ERROR"


class TestPipelineExceptions:
    def test_pipeline_error_node_attribute(self) -> None:
        err = PipelineError("fail", node="my_node")
        assert err.node == "my_node"
        assert err.code == "PIPELINE_ERROR"

    def test_intent_classification_error(self) -> None:
        err = IntentClassificationError("cannot classify")
        assert err.node == "intent_classifier"
        assert err.code == "INTENT_ERROR"
        assert isinstance(err, PipelineError)

    def test_entity_resolution_error_unresolved(self) -> None:
        err = EntityResolutionError("unresolved", unresolved_entities=["Alice", "Bob"])
        assert err.unresolved_entities == ["Alice", "Bob"]
        assert err.code == "ENTITY_RESOLUTION_ERROR"

    def test_entity_resolution_error_default_empty_list(self) -> None:
        err = EntityResolutionError("unresolved")
        assert err.unresolved_entities == []

    def test_cypher_generation_error(self) -> None:
        err = CypherGenerationError("bad cypher", generated_query="MATCH ???")
        assert err._generated_query == "MATCH ???"
        assert err.code == "CYPHER_GENERATION_ERROR"

    def test_empty_result_error_default_message(self) -> None:
        err = EmptyResultError()
        assert err.code == "EMPTY_RESULT"
        assert "No results" in err.message


class TestValidationExceptions:
    def test_validation_error_field_attribute(self) -> None:
        err = ValidationError("bad input", field="email")
        assert err.field == "email"
        assert err.code == "VALIDATION_ERROR"
        assert isinstance(err, GraphRAGError)

    def test_invalid_query_error_field(self) -> None:
        err = InvalidQueryError("empty question")
        assert err.field == "question"
        assert isinstance(err, ValidationError)


class TestMiscExceptions:
    def test_configuration_error(self) -> None:
        err = ConfigurationError("missing key", config_key="NEO4J_PASSWORD")
        assert err.config_key == "NEO4J_PASSWORD"
        assert err.code == "CONFIGURATION_ERROR"

    def test_conflict_error_versions(self) -> None:
        err = ConflictError("version mismatch", expected_version=3, current_version=5)
        assert err.expected_version == 3
        assert err.current_version == 5
        assert err.code == "VERSION_CONFLICT"

    def test_invalid_state_error(self) -> None:
        err = InvalidStateError("bad transition", current_state="PROCESSING")
        assert err.current_state == "PROCESSING"
        assert err.code == "INVALID_STATE"

    def test_authentication_error_default_message(self) -> None:
        err = AuthenticationError()
        assert err.code == "AUTH_REQUIRED"
        assert "Authentication" in err.message

    def test_authorization_error_default_message(self) -> None:
        err = AuthorizationError()
        assert err.code == "FORBIDDEN"
        assert "Permission" in err.message
