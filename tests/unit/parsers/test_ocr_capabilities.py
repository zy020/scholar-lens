from scholar_lens.parsers.ocr_capabilities import detect_rapidocr_capability


def test_detect_rapidocr_uninstalled(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    capability = detect_rapidocr_capability(vision_available=True)

    assert capability.engine == "rapidocr"
    assert capability.installed is False
    assert capability.gpu_available is False
    assert capability.cpu_available is False
    assert capability.recommended_mode == "vision_only"
    assert capability.available_actions == ["vision"]


def test_detect_rapidocr_disallows_cpu_when_gpu_unavailable(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "rapidocr" else None,
    )

    capability = detect_rapidocr_capability(vision_available=True, gpu_available=False)

    assert capability.installed is True
    assert capability.gpu_available is False
    assert capability.cpu_available is False
    assert capability.recommended_mode == "vision_only"
    assert capability.available_actions == ["vision"]


def test_detect_rapidocr_gpu_available(monkeypatch):
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name == "rapidocr" else None,
    )

    capability = detect_rapidocr_capability(vision_available=False, gpu_available=True)

    assert capability.recommended_mode == "gpu_ocr"
    assert capability.available_actions == ["gpu_ocr"]


def test_detect_rapidocr_gpu_provider_from_onnxruntime(monkeypatch):
    class FakeOnnxRuntime:
        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: object() if name in {"rapidocr", "onnxruntime"} else None,
    )
    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOnnxRuntime)

    capability = detect_rapidocr_capability()

    assert capability.gpu_available is True
    assert capability.recommended_mode == "gpu_ocr"
