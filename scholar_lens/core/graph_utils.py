from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

logger = logging.getLogger("scholar_lens.graph")


class GraphNodeError(RuntimeError):
    def __init__(self, graph_name: str, node_name: str, original: Exception) -> None:
        self.graph_name = graph_name
        self.node_name = node_name
        self.original = original
        super().__init__(f"{graph_name}.{node_name} failed: {original}")


async def trace_graph_node(
    graph_name: str,
    node_name: str,
    *,
    doc_id: str = "",
    func: Callable[[], Awaitable[Any]],
) -> Any:
    started = time.perf_counter()
    try:
        result = await func()
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "graph node completed",
            extra={
                "graph_name": graph_name,
                "node_name": node_name,
                "graph_node": f"{graph_name}.{node_name}",
                "doc_id": doc_id,
                "duration_ms": duration_ms,
                "status": "success",
            },
        )
        return result
    except GraphNodeError:
        raise
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "graph node failed",
            extra={
                "graph_name": graph_name,
                "node_name": node_name,
                "graph_node": f"{graph_name}.{node_name}",
                "doc_id": doc_id,
                "duration_ms": duration_ms,
                "status": "error",
                "error": str(exc),
            },
            exc_info=True,
        )
        raise GraphNodeError(graph_name, node_name, exc) from exc


async def invoke_llm_with_retries(
    llm,
    messages,
    *,
    graph_name: str,
    node_name: str,
    attempts: int = 2,
    timeout: float | None = None,
):
    last_error: Exception | None = None
    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        try:
            call = llm.ainvoke(messages)
            if timeout:
                return await asyncio.wait_for(call, timeout=timeout)
            return await call
        except Exception as exc:
            last_error = exc
            logger.warning(
                "LLM node call failed",
                extra={
                    "graph_name": graph_name,
                    "node_name": node_name,
                    "graph_node": f"{graph_name}.{node_name}",
                    "attempt": attempt,
                    "attempts": total_attempts,
                    "error": str(exc),
                },
            )
            if attempt >= total_attempts:
                break
            await asyncio.sleep(min(0.2 * attempt, 1.0))
    assert last_error is not None
    raise last_error
