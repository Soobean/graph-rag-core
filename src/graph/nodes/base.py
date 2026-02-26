"""Base Node Abstract Class"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from src.graph.state import GraphRAGState

DEFAULT_TIMEOUT = 30
DB_TIMEOUT = 15
CPU_TIMEOUT = 10


class BaseNode[T](ABC):
    """모든 LangGraph 노드가 상속할 추상 클래스 (타임아웃 포함)"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def input_keys(self) -> list[str]: ...

    @property
    def timeout_seconds(self) -> float:
        return DEFAULT_TIMEOUT

    @abstractmethod
    async def _process(self, state: GraphRAGState) -> T: ...

    async def __call__(self, state: GraphRAGState) -> T | dict[str, Any]:
        try:
            return await asyncio.wait_for(self._process(state), timeout=self.timeout_seconds)
        except TimeoutError:
            self._logger.error(f"Node '{self.name}' timed out after {self.timeout_seconds}s")
            return {
                "error": f"처리 시간이 초과되었습니다 ({self.name}: {self.timeout_seconds}초)",
                "execution_path": [f"{self.name}_timeout"],
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
