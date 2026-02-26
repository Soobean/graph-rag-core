"""Checkpointer 팩토리."""

import aiosqlite

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


async def create_checkpointer(db_path: str = ":memory:") -> BaseCheckpointSaver:
    """Checkpointer 인스턴스 생성.

    ':memory:'면 MemorySaver, 그 외에는 AsyncSqliteSaver 사용.
    """
    if db_path == ":memory:":
        return MemorySaver()
    conn = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver
