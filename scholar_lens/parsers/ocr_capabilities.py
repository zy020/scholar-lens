from __future__ import annotations

import importlib.util

from pydantic import BaseModel, Field


class OCRCapability(BaseModel):
    engine: str = "rapidocr"
    installed: bool = False
    gpu_available: bool = False
    cpu_available: bool = False
    recommended_mode: str = "unavailable"
    available_actions: list[str] = Field(default_factory=list)


def detect_rapidocr_capability(
    vision_available: bool = False,
    gpu_available: bool | None = None,
) -> OCRCapability:
    installed = _rapidocr_installed()
    detected_gpu = _onnx_cuda_available() if gpu_available is None else gpu_available
    gpu_ok = bool(installed and detected_gpu is True)
    cpu_ok = False

    actions: list[str] = []
    recommended_mode = "unavailable"

    if gpu_ok:
        actions.append("gpu_ocr")
        recommended_mode = "gpu_ocr"
    elif installed and vision_available:
        recommended_mode = "vision_only"

    if vision_available:
        actions.append("vision")
        if not installed:
            recommended_mode = "vision_only"

    return OCRCapability(
        installed=installed,
        gpu_available=gpu_ok,
        cpu_available=cpu_ok,
        recommended_mode=recommended_mode,
        available_actions=actions,
    )


def _rapidocr_installed() -> bool:
    return importlib.util.find_spec("rapidocr") is not None


def _onnx_cuda_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    try:
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False
