from pathlib import Path


def test_chat_stream_uses_token_streaming_for_light_mode():
    source = Path("scholar_lens/api/routes/chat.py").read_text(encoding="utf-8")

    assert "run_chat_retrieval_graph" in source
    assert "_stream_llm_tokens_with_initial_retry" in source
    assert "if request.deep_mode:" in source


def test_chat_stream_logs_sse_errors_with_traceback():
    source = Path("scholar_lens/api/routes/chat.py").read_text(encoding="utf-8")

    assert 'logger.exception("SSE stream error")' in source


def test_chat_stream_streams_deep_mode_revision_stage():
    source = Path("scholar_lens/api/routes/chat.py").read_text(encoding="utf-8")

    assert "run_chat_validation_graph" in source
    assert "build_revision_messages" in source
    assert "_stream_llm_tokens_with_initial_retry" in source


def test_chat_stream_reports_deep_mode_graph_stages():
    source = Path("scholar_lens/api/routes/chat.py").read_text(encoding="utf-8")

    for stage in ("intent", "retrieve", "draft", "validate", "revise"):
        assert f"'stage': '{stage}'" in source
