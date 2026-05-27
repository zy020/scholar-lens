from pathlib import Path

import pytest

from scholar_lens.core.settings import VisionConfig
from scholar_lens.parsers.vision_executor import VisionEnhancementExecutor, VisionUnavailableError, parse_vision_structured_content


def test_vision_executor_sends_images_and_parses_response(tmp_path, monkeypatch):
    source = tmp_path / "slides.pdf"
    source.write_bytes(b"%PDF-1.4")
    image = tmp_path / "page_2.png"
    image.write_bytes(b"png-bytes")
    calls = []

    def fake_prepare(source_path: Path, pages: list[int], work_dir: Path):
        calls.append(("prepare", source_path, pages))
        return {2: [image]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(("post", url, headers, json, timeout))

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "A diagram explaining attention weights."
                            }
                        }
                    ]
                }

        return Response()

    monkeypatch.setattr("scholar_lens.parsers.vision_executor.prepare_vision_images", fake_prepare)
    executor = VisionEnhancementExecutor(
        config=VisionConfig(api_key="test-api-key", base_url="https://vision.example/v1", model="vision-model"),
        post=fake_post,
    )

    result = executor.run(source, pages=[2])

    assert result.status == "completed"
    assert result.pages[0].page == 2
    assert result.pages[0].text == "A diagram explaining attention weights."
    assert calls[0] == ("prepare", source, [2])
    request = calls[1][3]
    assert request["model"] == "vision-model"
    content = request["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_vision_executor_parses_structured_visual_json(tmp_path, monkeypatch):
    source = tmp_path / "slides.pdf"
    source.write_bytes(b"%PDF-1.4")
    image = tmp_path / "page_4.png"
    image.write_bytes(b"png-bytes")

    def fake_prepare(source_path: Path, pages: list[int], work_dir: Path):
        return {4: [image]}

    def fake_post(url, headers=None, json=None, timeout=None):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": """```json
{
  "text": "The slide defines scaled dot-product attention.",
  "visual_type": "formula",
  "key_observations": ["Q and K are multiplied", "sqrt(d_k) normalizes scores"],
  "formula_summary": "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
  "table_summary": "",
  "chart_summary": "",
  "qa_hint": "Useful for questions about the attention formula."
}
```"""
                            }
                        }
                    ]
                }

        return Response()

    monkeypatch.setattr("scholar_lens.parsers.vision_executor.prepare_vision_images", fake_prepare)
    executor = VisionEnhancementExecutor(
        config=VisionConfig(api_key="test-api-key", base_url="https://vision.example/v1", model="vision-model"),
        post=fake_post,
    )

    result = executor.run(source, pages=[4])

    page = result.pages[0]
    assert page.visual_type == "formula"
    assert page.formula_summary.startswith("Attention")
    assert page.key_observations == ["Q and K are multiplied", "sqrt(d_k) normalizes scores"]
    assert "Visual type: formula" in page.text


def test_parse_vision_structured_content_falls_back_to_text_for_plain_markdown():
    parsed = parse_vision_structured_content("A diagram explaining attention weights.")

    assert parsed["text"] == "A diagram explaining attention weights."
    assert parsed["visual_type"] == "mixed"


def test_parse_vision_structured_content_keeps_empty_json_unusable():
    parsed = parse_vision_structured_content(
        '{"text": "", "visual_type": "mixed", "key_observations": [], '
        '"formula_summary": "", "table_summary": "", "chart_summary": "", "qa_hint": ""}'
    )

    assert parsed["text"] == ""
    assert parsed["visual_type"] == "mixed"


def test_vision_executor_rejects_pptx_source(tmp_path):
    source = tmp_path / "slides.pptx"
    source.write_bytes(b"pptx")

    executor = VisionEnhancementExecutor(
        config=VisionConfig(api_key="test-api-key", base_url="https://vision.example/v1", model="vision-model"),
        post=lambda *args, **kwargs: None,
    )

    with pytest.raises(VisionUnavailableError, match="PDF sources only"):
        executor.run(source, pages=[1])
