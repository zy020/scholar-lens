from pathlib import Path
import sys
import types

import pytest

from scholar_lens.parsers.ocr_executor import (
    OCRUnavailableError,
    RapidOCRExecutor,
    normalize_rapidocr_output,
)


def test_normalize_rapidocr_output_accepts_common_tuple_shape():
    output = ([[[0, 0], [1, 0]], "Hello", 0.99], [[[0, 1], [1, 1]], "World", 0.95])

    assert normalize_rapidocr_output(output) == "Hello\nWorld"


def test_normalize_rapidocr_output_accepts_nested_result_shape():
    output = ([[[[0, 0], [1, 0]], "Slide", 0.9], [[[0, 1], [1, 1]], "Title", 0.8]], 0.05)

    assert normalize_rapidocr_output(output) == "Slide\nTitle"


def test_executor_rejects_non_pdf_source(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("not supported", encoding="utf-8")
    executor = RapidOCRExecutor(ocr_callable=lambda image: "ignored")

    with pytest.raises(OCRUnavailableError):
        executor.run(source, pages=[0])


def test_executor_rejects_gpu_ocr_when_cuda_provider_unavailable(monkeypatch):
    monkeypatch.setattr("scholar_lens.parsers.ocr_executor.ocr_gpu_available", lambda: False)
    executor = RapidOCRExecutor()

    with pytest.raises(OCRUnavailableError, match="GPU OCR"):
        executor._load_ocr_callable()


def test_executor_loads_rapidocr_with_cuda_params(monkeypatch):
    captured = {}

    class FakeRapidOCR:
        def __init__(self, params=None):
            captured["params"] = params

    fake_module = types.SimpleNamespace(RapidOCR=FakeRapidOCR)
    monkeypatch.setattr("scholar_lens.parsers.ocr_executor.ocr_gpu_available", lambda: True)
    monkeypatch.setitem(sys.modules, "rapidocr", fake_module)

    executor = RapidOCRExecutor()
    ocr = executor._load_ocr_callable()

    assert isinstance(ocr, FakeRapidOCR)
    assert captured["params"] == {"EngineConfig.onnxruntime.use_cuda": True}


def test_executor_runs_fake_ocr_and_evaluates_quality(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.4")
    rendered = []

    def fake_render(source_path: Path, pages: list[int], work_dir: Path):
        rendered.extend(pages)
        image = work_dir / "page_2.png"
        image.write_bytes(b"png")
        return {2: [image]}

    monkeypatch.setattr("scholar_lens.parsers.ocr_executor.render_pdf_pages", fake_render)
    executor = RapidOCRExecutor(ocr_callable=lambda image: [[None, "Recognized lecture text with enough detail", 0.9]])

    result = executor.run(source, pages=[2])

    assert rendered == [2]
    assert result.pages[0].page == 2
    assert result.pages[0].ocr_quality == "good"
    assert result.pages[0].vision_recommended is False


def test_executor_rejects_pptx_source(tmp_path):
    source = tmp_path / "source.pptx"
    source.write_bytes(b"pptx")
    executor = RapidOCRExecutor(ocr_callable=lambda image: "ignored")

    with pytest.raises(OCRUnavailableError, match="PDF sources only"):
        executor.run(source, pages=[1])
