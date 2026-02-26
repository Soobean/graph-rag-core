"""LLM Repository - Azure OpenAI 접근 계층 (openai SDK 직접 사용)"""

import json
import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, cast

from openai import APIConnectionError, APIStatusError, AsyncAzureOpenAI, RateLimitError

from src.config import Settings
from src.domain.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from src.domain.types import (
    CypherGenerationResult,
    EntityExtractionResult,
    IntentClassificationResult,
    IntentEntityExtractionResult,
    QueryDecompositionResult,
)
from src.utils.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

FALLBACK_EXCEPTIONS = (LLMRateLimitError, LLMConnectionError, LLMResponseError)


class ModelTier(str, Enum):
    LIGHT = "light"
    HEAVY = "heavy"


class LLMRepository:
    """Azure OpenAI LLM Repository"""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: AsyncAzureOpenAI | None = None
        self._prompt_manager = PromptManager()

    def _get_client(self) -> AsyncAzureOpenAI:
        if self._client is None:
            try:
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self._settings.azure_openai_endpoint,
                    api_key=self._settings.azure_openai_api_key,
                    api_version=self._settings.azure_openai_api_version,
                    timeout=60.0,
                    max_retries=3,
                )
            except Exception as e:
                raise LLMConnectionError(f"Failed to initialize LLM client: {e}") from e
        return self._client

    def _get_deployment(self, tier: ModelTier) -> str:
        if tier == ModelTier.LIGHT:
            return self._settings.light_model_deployment
        return self._settings.heavy_model_deployment

    # ── Core API call ──────────────────────────────────────

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model_tier: ModelTier = ModelTier.LIGHT,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """공통 LLM API 호출"""
        client = self._get_client()
        deployment = self._get_deployment(model_tier)

        api_params: dict[str, Any] = {
            "model": deployment,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": max_completion_tokens or self._settings.llm_max_tokens,
        }

        if json_mode:
            api_params["response_format"] = {"type": "json_object"}

        if not deployment.lower().startswith("gpt-5"):
            api_params["temperature"] = temperature if temperature is not None else self._settings.llm_temperature

        try:
            response = await client.chat.completions.create(**api_params)
            if not response.choices:
                raise LLMResponseError("No response choices returned from LLM")
            content = response.choices[0].message.content
            if not content or not content.strip():
                if json_mode:
                    raise LLMResponseError("Empty JSON response from LLM")
                return ""
            return content
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except APIStatusError as e:
            raise LLMResponseError(f"API error: {e.status_code} - {e.message}") from e
        except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Failed to generate response: {e}") from e

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model_tier: ModelTier = ModelTier.LIGHT,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
    ) -> str:
        return await self._call_api(system_prompt, user_prompt, model_tier, temperature, max_completion_tokens)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_tier: ModelTier = ModelTier.LIGHT,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        content = await self._call_api(system_prompt, user_prompt, model_tier, temperature, json_mode=True)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Invalid JSON response: {e}") from e

    # ── Fallback (HEAVY → LIGHT) ───────────────────────────

    async def _with_fallback(
        self,
        fn_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
    ) -> str | dict[str, Any]:
        """HEAVY → LIGHT fallback"""
        fn = getattr(self, fn_name)
        try:
            return await fn(system_prompt=system_prompt, user_prompt=user_prompt,
                            model_tier=ModelTier.HEAVY, temperature=temperature,
                            **({"max_completion_tokens": max_completion_tokens} if max_completion_tokens and fn_name == "generate" else {}))
        except FALLBACK_EXCEPTIONS as e:
            logger.warning(f"HEAVY tier failed, falling back to LIGHT: {e}")

        try:
            return await fn(system_prompt=system_prompt, user_prompt=user_prompt,
                            model_tier=ModelTier.LIGHT, temperature=temperature,
                            **({"max_completion_tokens": max_completion_tokens} if max_completion_tokens and fn_name == "generate" else {}))
        except FALLBACK_EXCEPTIONS as e:
            raise LLMResponseError(f"All model tiers failed. Last error: {e}") from e

    async def _generate_with_fallback(self, system_prompt: str, user_prompt: str,
                                       temperature: float | None = None, max_completion_tokens: int | None = None) -> str:
        return cast(str, await self._with_fallback("generate", system_prompt, user_prompt, temperature, max_completion_tokens))

    async def _generate_json_with_fallback(self, system_prompt: str, user_prompt: str,
                                            temperature: float | None = None) -> dict[str, Any]:
        return cast(dict[str, Any], await self._with_fallback("generate_json", system_prompt, user_prompt, temperature))

    # ── Domain methods ─────────────────────────────────────

    def _chat_history(self, chat_history: str) -> str:
        return chat_history.strip() or "(No previous conversation)"

    async def classify_intent(self, question: str, available_intents: list[str], chat_history: str = "") -> IntentClassificationResult:
        prompt = self._prompt_manager.load_prompt("intent_classification")
        result = await self.generate_json(
            system_prompt=prompt["system"].format(available_intents=", ".join(available_intents)),
            user_prompt=prompt["user"].format(question=question, chat_history=self._chat_history(chat_history)),
            model_tier=ModelTier.LIGHT,
        )
        return cast(IntentClassificationResult, result)

    async def extract_entities(self, question: str, entity_types: list[str], chat_history: str = "") -> EntityExtractionResult:
        prompt = self._prompt_manager.load_prompt("entity_extraction")
        result = await self.generate_json(
            system_prompt=prompt["system"].format(entity_types=", ".join(entity_types)),
            user_prompt=prompt["user"].format(question=question, chat_history=self._chat_history(chat_history)),
            model_tier=ModelTier.LIGHT,
        )
        return cast(EntityExtractionResult, result)

    async def classify_intent_and_extract_entities(
        self, question: str, available_intents: list[str], entity_types: list[str],
        chat_history: str = "", schema_summary: str = "",
    ) -> IntentEntityExtractionResult:
        prompt = self._prompt_manager.load_prompt("intent_entity_combined")
        result = await self.generate_json(
            system_prompt=prompt["system"].format(
                available_intents=", ".join(available_intents),
                entity_types=", ".join(entity_types),
                schema_summary=schema_summary or "Schema information not available",
            ),
            user_prompt=prompt["user"].format(question=question, chat_history=self._chat_history(chat_history)),
            model_tier=ModelTier.LIGHT,
        )
        return cast(IntentEntityExtractionResult, result)

    async def generate_cypher(
        self, question: str, schema: dict[str, Any], entities: list[dict[str, Any]],
        query_plan: dict[str, Any] | None = None, use_light_model: bool = False, intent: str = "",
    ) -> CypherGenerationResult:
        prompt = self._prompt_manager.load_prompt("cypher_generation")
        system_prompt = prompt["system"].format(schema_str=self._format_schema(schema))
        user_prompt = prompt["user"].format(
            question=question,
            entities_str=self._format_entities(entities),
            query_plan_str=self._format_query_plan(query_plan),
            intent=intent or "unknown",
        )

        if use_light_model:
            result = await self.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, model_tier=ModelTier.LIGHT)
        else:
            result = await self._generate_json_with_fallback(system_prompt=system_prompt, user_prompt=user_prompt)
        return cast(CypherGenerationResult, result)

    async def generate_response(self, question: str, query_results: list[dict[str, Any]],
                                 cypher_query: str, chat_history: str = "") -> str:
        prompt = self._prompt_manager.load_prompt("response_generation")
        return await self._generate_with_fallback(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"].format(
                question=question,
                results_str=self._format_results(query_results),
                chat_history=self._chat_history(chat_history),
            ),
        )

    async def generate_response_stream(self, question: str, query_results: list[dict[str, Any]],
                                        cypher_query: str, chat_history: str = "") -> AsyncIterator[str]:
        prompt = self._prompt_manager.load_prompt("response_generation")
        client = self._get_client()
        deployment = self._get_deployment(ModelTier.HEAVY)

        api_params: dict[str, Any] = {
            "model": deployment,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"].format(
                    question=question,
                    results_str=self._format_results(query_results),
                    chat_history=self._chat_history(chat_history),
                )},
            ],
            "max_completion_tokens": self._settings.llm_max_tokens,
            "stream": True,
        }

        if not deployment.lower().startswith("gpt-5"):
            api_params["temperature"] = self._settings.llm_temperature

        try:
            response = await client.chat.completions.create(**api_params)
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except APIStatusError as e:
            raise LLMResponseError(f"API error: {e.status_code} - {e.message}") from e
        except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Failed to stream response: {e}") from e

    async def generate_clarification(self, question: str, unresolved_entities: str) -> str:
        prompt = self._prompt_manager.load_prompt("clarification")
        return await self.generate(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"].format(question=question, unresolved_entities=unresolved_entities or "없음"),
            model_tier=ModelTier.LIGHT,
        )

    async def decompose_query(self, question: str, schema: dict[str, Any] | None = None) -> QueryDecompositionResult:
        prompt = self._prompt_manager.load_prompt("query_decomposition")
        schema_str = self._format_schema(schema) if schema else "Schema information not available"
        result = await self.generate_json(
            system_prompt=prompt["system"].format(schema_str=schema_str),
            user_prompt=prompt["user"].format(question=question),
            model_tier=ModelTier.LIGHT,
        )
        return cast(QueryDecompositionResult, result)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ── Embedding ──────────────────────────────────────────

    async def get_embedding(self, text: str) -> list[float]:
        client = self._get_client()
        try:
            response = await client.embeddings.create(
                model=self._settings.embedding_model_deployment,
                input=text,
                dimensions=self._settings.embedding_dimensions,
            )
            return response.data[0].embedding
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except APIStatusError as e:
            raise LLMResponseError(f"Embedding API error: {e.status_code} - {e.message}") from e
        except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Failed to generate embedding: {e}") from e

    async def get_embeddings_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        deployment = self._settings.embedding_model_deployment
        all_embeddings: list[list[float]] = []

        try:
            for i in range(0, len(texts), batch_size):
                response = await client.embeddings.create(
                    model=deployment,
                    input=texts[i : i + batch_size],
                    dimensions=self._settings.embedding_dimensions,
                )
                all_embeddings.extend(item.embedding for item in response.data)
            return all_embeddings
        except RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except APIConnectionError as e:
            raise LLMConnectionError(str(e)) from e
        except APIStatusError as e:
            raise LLMResponseError(f"Embedding API error: {e.status_code} - {e.message}") from e
        except (LLMRateLimitError, LLMConnectionError, LLMResponseError):
            raise
        except Exception as e:
            raise LLMResponseError(f"Failed to generate batch embeddings: {e}") from e

    # ── Formatters ─────────────────────────────────────────

    def _format_schema(self, schema: dict[str, Any]) -> str:
        lines: list[str] = []

        def _format_props(props: list[dict[str, Any]]) -> str:
            parts = []
            for p in props:
                name = p.get("name", "")
                if not name:
                    continue
                sample = p.get("sample_values")
                parts.append(f"{name}[{', '.join(sample)}]" if sample else name)
            return ", ".join(parts)

        nodes = schema.get("nodes")
        if nodes:
            lines.append("Nodes:")
            for node in nodes:
                label = node.get("label", "Unknown")
                props = node.get("properties", [])
                formatted = _format_props(props)
                lines.append(f"  {label} ({formatted})" if formatted else f"  {label}")
        else:
            labels = schema.get("node_labels", [])
            if labels:
                lines.append(f"Node Labels: {', '.join(labels)}")

        rels = schema.get("relationships")
        if rels:
            lines.append("Relationships:")
            for rel in rels:
                rel_type = rel.get("type", "Unknown")
                props = rel.get("properties", [])
                formatted = _format_props(props)
                lines.append(f"  {rel_type} ({formatted})" if formatted else f"  {rel_type}")
        else:
            rel_types = schema.get("relationship_types", [])
            if rel_types:
                lines.append(f"Relationship Types: {', '.join(rel_types)}")

        return "\n".join(lines) if lines else "Schema information not available"

    def _format_entities(self, entities: list[dict[str, Any]]) -> str:
        if not entities:
            return "No entities extracted"
        return "\n".join(
            f"- {e.get('type', 'Unknown')}: {e.get('value', '')} (normalized: {e.get('normalized', '')})"
            for e in entities
        )

    def _format_query_plan(self, query_plan: dict[str, Any] | None) -> str:
        if not query_plan:
            return "No query plan (single-hop query)"
        if not query_plan.get("is_multi_hop"):
            return "Single-hop query"

        lines = [
            f"Multi-hop Query Plan ({query_plan.get('hop_count', 0)} hops):",
            f"Goal: {query_plan.get('final_return', 'unknown')}",
        ]
        for hop in query_plan.get("hops", []):
            hop_line = f"  Step {hop.get('step', '?')}: {hop.get('description', '')}"
            if rel := hop.get("relationship", ""):
                hop_line += f" [{rel}, {hop.get('direction', '')}]"
            if filter_cond := hop.get("filter_condition", ""):
                hop_line += f" WHERE {filter_cond}"
            lines.append(hop_line)
        return "\n".join(lines)

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "No results found"

        entities: dict[str, dict[str, Any]] = {}
        scalar_rows: list[dict[str, Any]] = []

        for row in results:
            has_node = False
            for value in row.values():
                if not isinstance(value, dict):
                    continue
                if "labels" in value and isinstance(value.get("labels"), list):
                    has_node = True
                    node_id = value.get("id", value.get("elementId", ""))
                    if node_id and node_id not in entities:
                        props = value.get("properties", {})
                        entities[node_id] = {
                            "label": value["labels"][0] if value["labels"] else "Node",
                            "name": props.get("name", "Unknown"),
                            "properties": props,
                        }
            if not has_node:
                scalar_rows.append(row)

        lines: list[str] = []

        if entities:
            by_label: dict[str, list[dict[str, Any]]] = {}
            for entity in entities.values():
                by_label.setdefault(entity["label"], []).append(entity)

            lines.append(f"총 {len(entities)}개의 고유 엔티티, {len(results)}개의 관계 결과:")
            for label, ents in by_label.items():
                lines.append(f"\n[{label}] ({len(ents)}개):")
                for i, ent in enumerate(ents[:15], 1):
                    props = {k: v for k, v in ent.get("properties", {}).items()
                             if k not in ("embedding", "vector") and v is not None}
                    lines.append(f"  {i}. {ent['name']} ({', '.join(f'{k}={v}' for k, v in list(props.items())[:5])})")
                if len(ents) > 15:
                    lines.append(f"  ... 외 {len(ents) - 15}개")

        if scalar_rows:
            if lines:
                lines.append("")
            lines.append(f"집계 결과 ({len(scalar_rows)}행):")
            for i, row in enumerate(scalar_rows[:20], 1):
                lines.append(f"  {i}. {', '.join(f'{k}={v}' for k, v in row.items() if v is not None)}")
            if len(scalar_rows) > 20:
                lines.append(f"  ... 외 {len(scalar_rows) - 20}개")

        return "\n".join(lines) if lines else "No results found"
