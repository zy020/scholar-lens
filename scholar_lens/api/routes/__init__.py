from scholar_lens.api.routes.chat import router as chat_router
from scholar_lens.api.routes.config import router as config_router
from scholar_lens.api.routes.documents import router as documents_router
from scholar_lens.api.routes.memory import router as memory_router
from scholar_lens.api.routes.notes import router as notes_router

__all__ = ["chat_router", "config_router", "documents_router", "memory_router", "notes_router"]
