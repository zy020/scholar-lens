from __future__ import annotations

import logging
import sys

GRAPH_LOGGER_NAME = "scholar_lens.graph"
STRUCTURED_LOGGER_NAMES = ("scholar_lens", GRAPH_LOGGER_NAME, "langchain", "langgraph", "httpx")


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    for logger_name in STRUCTURED_LOGGER_NAMES:
        lg = logging.getLogger(logger_name)
        lg.addHandler(handler)
        lg.setLevel(level if logger_name in {"scholar_lens", GRAPH_LOGGER_NAME} else logging.WARNING)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
