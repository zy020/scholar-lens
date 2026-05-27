from pathlib import Path

from scholar_lens.core.settings import VisionConfig
from scholar_lens.parsers.vision_executor import VisionEnhancementExecutor


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


def test_vision_executor_marks_pptx_slide_without_images(tmp_path, monkeypatch):
    source = tmp_path / "slides.pptx"
    source.write_bytes(b"pptx")

    monkeypatch.setattr("scholar_lens.parsers.vision_executor.prepare_vision_images", lambda source, pages, work_dir: {})
    executor = VisionEnhancementExecutor(
        config=VisionConfig(api_key="test-api-key", base_url="https://vision.example/v1", model="vision-model"),
        post=lambda *args, **kwargs: None,
    )

    result = executor.run(source, pages=[1])

    assert result.status == "failed"
    assert result.pages[0].page == 1
    assert result.pages[0].reason == "pptx_no_embedded_images"
    assert "embedded images" in result.pages[0].error
