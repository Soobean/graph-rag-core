"""
Cypher Generator Node

사용자 질문과 엔티티 정보를 바탕으로 Cypher 쿼리를 생성합니다.
캐시 히트 시에는 생성을 스킵합니다.

Latency Optimization:
- 단순 쿼리(single-hop, 기본 의도, 적은 엔티티)에는 LIGHT 모델 사용
- 복잡한 쿼리(multi-hop, 복잡한 의도)에는 HEAVY 모델 사용
"""

import re
from enum import Enum
from typing import Any

from src.config import Settings
from src.domain.types import CypherGeneratorUpdate, GraphSchema
from src.graph.nodes.base import BaseNode
from src.graph.state import GraphRAGState
from src.repositories.llm_repository import LLMRepository
from src.repositories.neo4j_repository import Neo4jRepository


class QueryComplexity(str, Enum):
    """쿼리 복잡도 구분"""

    SIMPLE = "simple"  # LIGHT 모델 사용
    COMPLEX = "complex"  # HEAVY 모델 사용 (fallback 포함)


class CypherGeneratorNode(BaseNode[CypherGeneratorUpdate]):
    """
    Cypher 쿼리 생성 노드

    Latency Optimization:
    - 단순 쿼리 판별 기준:
      1. Single-hop 쿼리 (is_multi_hop=False)
      2. 단순 Intent (SIMPLE_INTENTS에 포함)
      3. 낮은 엔티티 수 (≤ 2개)
    - 모든 조건 충족 시 LIGHT 모델 사용으로 ~400ms 절감
    """

    # 단순 Intent 목록 (LIGHT 모델로 처리 가능) — 도메인별 커스터마이징
    SIMPLE_INTENTS = ["entity_search"]

    # 단순 쿼리의 최대 엔티티 수
    MAX_ENTITIES_FOR_SIMPLE = 2

    def __init__(
        self,
        llm_repository: LLMRepository,
        neo4j_repository: Neo4jRepository,
        settings: Settings | None = None,
    ):
        super().__init__()
        self._llm = llm_repository
        self._neo4j = neo4j_repository
        self._settings = settings
        self._schema_cache: GraphSchema | None = None

    @property
    def name(self) -> str:
        return "cypher_generator"

    @property
    def input_keys(self) -> list[str]:
        return ["question", "entities"]

    async def _get_schema(self) -> GraphSchema:
        """스키마 정보 조회 (캐싱)"""
        if self._schema_cache is None:
            schema_dict = await self._neo4j.get_schema()
            self._schema_cache = GraphSchema(
                node_labels=schema_dict.get("node_labels", []),
                relationship_types=schema_dict.get("relationship_types", []),
                nodes=schema_dict.get("nodes", []),
                relationships=schema_dict.get("relationships", []),
                indexes=schema_dict.get("indexes", []),
                constraints=schema_dict.get("constraints", []),
            )
        return self._schema_cache

    def _analyze_complexity(self, state: GraphRAGState) -> QueryComplexity:
        """쿼리 복잡도 분석"""
        query_plan = state.get("query_plan")
        intent = state.get("intent", "unknown")
        entities = state.get("entities", {})

        if query_plan and query_plan.get("is_multi_hop"):
            self._logger.debug(
                f"Complex query: multi-hop detected (hops={query_plan.get('hop_count')})"
            )
            return QueryComplexity.COMPLEX

        if intent not in self.SIMPLE_INTENTS:
            self._logger.debug(f"Complex query: complex intent ({intent})")
            return QueryComplexity.COMPLEX

        total_entities = sum(len(v) for v in entities.values())
        if total_entities > self.MAX_ENTITIES_FOR_SIMPLE:
            self._logger.debug(
                f"Complex query: too many entities ({total_entities} > {self.MAX_ENTITIES_FOR_SIMPLE})"
            )
            return QueryComplexity.COMPLEX

        self._logger.debug(f"Simple query: intent={intent}, entities={total_entities}")
        return QueryComplexity.SIMPLE

    def _correct_single_value(
        self,
        value: str,
        entity_values: list[str],
    ) -> str:
        """단일 문자열 값을 엔티티 값으로 보정."""
        exact = next(
            (ev for ev in entity_values if ev.lower() == value.lower()),
            None,
        )
        if exact is not None:
            return exact

        val_lower = value.lower()
        contains_matches = [ev for ev in entity_values if ev.lower() in val_lower]
        if contains_matches:
            best = max(contains_matches, key=len)
            self._logger.info(
                f"Parameter correction: '{value}' → '{best}' (param contains entity)"
            )
            return best

        reverse_matches = [ev for ev in entity_values if val_lower in ev.lower()]
        if reverse_matches:
            best = min(reverse_matches, key=len)
            self._logger.info(
                f"Parameter correction: '{value}' → '{best}' (entity contains param)"
            )
            return best

        return value

    def _correct_parameters(
        self,
        parameters: dict[str, Any],
        entities: dict[str, list[str]],
    ) -> dict[str, Any]:
        """LLM이 생성한 파라미터 값을 엔티티 값으로 보정."""
        entity_values: list[str] = []
        for values in entities.values():
            entity_values.extend(values)

        if not entity_values:
            return parameters

        corrected: dict[str, Any] = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                corrected[key] = self._correct_single_value(value, entity_values)
            elif isinstance(value, list):
                corrected[key] = [
                    self._correct_single_value(v, entity_values)
                    if isinstance(v, str)
                    else v
                    for v in value
                ]
            else:
                corrected[key] = value

        return corrected

    def _fix_not_in_syntax(self, cypher: str) -> str:
        """Neo4j Cypher NOT IN 문법 보정"""
        fixed = re.sub(
            r"\b(WHERE|AND|OR)\s+"
            r"((?:\w+\s*\([^)]*\)|\w+(?:\.\w+)*))"
            r"\s+NOT\s+IN\b",
            r"\1 NOT \2 IN",
            cypher,
            flags=re.IGNORECASE,
        )
        if fixed != cypher:
            self._logger.info("Fixed NOT IN syntax in Cypher query")
        return fixed

    def _fix_aggregation_type_a_return(self, cypher: str) -> str:
        """WITH + 집계 후 re-MATCH + TYPE A RETURN 안티패턴을 TYPE B로 변환."""
        lines = [line.strip() for line in cypher.strip().split("\n") if line.strip()]

        agg_with_idx = -1
        agg_funcs = ("COUNT(", "SUM(", "AVG(", "COLLECT(")
        for i, line in enumerate(lines):
            upper = line.upper()
            if upper.startswith("WITH") and any(f in upper for f in agg_funcs):
                agg_with_idx = i
                break

        if agg_with_idx < 0:
            return cypher

        re_match_idx = -1
        for i in range(agg_with_idx + 1, len(lines)):
            if lines[i].upper().startswith("MATCH"):
                re_match_idx = i
                break

        if re_match_idx < 0:
            return cypher

        return_idx = -1
        for i in range(re_match_idx, len(lines)):
            if lines[i].upper().startswith("RETURN"):
                return_idx = i
                break

        if return_idx < 0 or " AS " in lines[return_idx].upper():
            return cypher

        with_line = lines[agg_with_idx]
        main_var_match = re.match(r"WITH\s+(\w+)\s*,", with_line, re.IGNORECASE)
        if not main_var_match:
            return cypher
        main_var = main_var_match.group(1)

        aliases = re.findall(r"\bAS\s+(\w+)", with_line, re.IGNORECASE)
        if not aliases:
            return cypher

        return_parts = [f"{main_var}.name AS name"] + aliases
        new_return = "RETURN " + ", ".join(return_parts)
        result_lines = lines[:re_match_idx] + [new_return, f"ORDER BY {aliases[0]} DESC"]
        fixed = "\n".join(result_lines)

        self._logger.info("Fixed aggregation + TYPE A return → TYPE B")
        return fixed

    def _coerce_tolower_params(
        self,
        cypher: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Cypher에서 toLower()에 사용되는 파라미터가 숫자 타입이면 문자열로 변환"""
        tolower_params: set[str] = set()
        tolower_params.update(re.findall(r"toLower\(\s*\$(\w+)\s*\)", cypher))

        for iter_var, param_name in re.findall(r"(\w+)\s+IN\s+\$(\w+)", cypher):
            if re.search(rf"toLower\(\s*{re.escape(iter_var)}\s*\)", cypher):
                tolower_params.add(param_name)

        if not tolower_params:
            return parameters

        coerced = dict(parameters)
        for param_name in tolower_params:
            if param_name in coerced:
                value = coerced[param_name]
                if isinstance(value, (int, float)):
                    self._logger.info(
                        f"Coercing toLower param '{param_name}': "
                        f"{type(value).__name__}({value}) → str('{value}')"
                    )
                    coerced[param_name] = str(value)
                elif isinstance(value, list):
                    coerced[param_name] = [
                        str(v) if isinstance(v, (int, float)) else v for v in value
                    ]
        return coerced

    async def _process(self, state: GraphRAGState) -> CypherGeneratorUpdate:
        """Cypher 쿼리 생성"""
        question = state.get("question", "")

        # 캐시 히트 시 스킵
        if state.get("skip_generation"):
            self._logger.info("Skipping Cypher generation (cache hit)")
            return CypherGeneratorUpdate(
                execution_path=[f"{self.name}_cached"],
            )

        raw_entities = state.get("entities", {})
        formatted_entities: list[dict[str, str]] = []
        for entity_type, values in raw_entities.items():
            for value in values:
                formatted_entities.append({"type": entity_type, "value": value})

        query_plan = state.get("query_plan")

        self._logger.info(f"Generating Cypher for: {question[:50]}...")

        try:
            # 스키마 정보 조회
            schema: GraphSchema
            state_schema = state.get("schema")
            if state_schema:
                schema = state_schema
            else:
                schema = await self._get_schema()

            # 복잡도 분석 및 모델 선택
            use_light_model = False
            if self._settings and self._settings.cypher_light_model_enabled:
                complexity = self._analyze_complexity(state)
                use_light_model = complexity == QueryComplexity.SIMPLE
                self._logger.info(
                    f"Query complexity: {complexity.value}, use_light_model={use_light_model}"
                )

            # LLM을 통한 Cypher 생성
            intent = state.get("intent", "unknown")
            result = await self._llm.generate_cypher(
                question=question,
                schema=dict(schema),
                entities=formatted_entities,
                query_plan=dict(query_plan) if query_plan else None,
                use_light_model=use_light_model,
                intent=intent,
            )

            cypher = result.get("cypher", "")
            parameters = result.get("parameters", {})

            # Cypher 문법 보정
            cypher = self._fix_not_in_syntax(cypher)
            cypher = self._fix_aggregation_type_a_return(cypher)

            # 파라미터를 엔티티 값으로 보정
            parameters = self._correct_parameters(parameters, raw_entities)
            parameters = self._coerce_tolower_params(cypher, parameters)

            if not cypher or not cypher.strip():
                raise ValueError("Empty Cypher query generated")

            self._logger.info(f"Generated Cypher: {cypher[:100]}...")
            self._logger.debug(f"Parameters: {parameters}")

            return CypherGeneratorUpdate(
                schema=schema,
                cypher_query=cypher,
                cypher_parameters=parameters,
                execution_path=[self.name],
            )

        except Exception as e:
            self._logger.error(f"Cypher generation failed: {e}")
            return CypherGeneratorUpdate(
                cypher_query="",
                cypher_parameters={},
                error=f"Cypher generation failed: {e}",
                execution_path=[f"{self.name}_error"],
            )
