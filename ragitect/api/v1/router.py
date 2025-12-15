"""API v1 router

Main router for API version 1 that includes all sub-routers.
"""

from fastapi import APIRouter

from ragitect.api.v1.chat import router as chat_router
from ragitect.api.v1.documents import documents_router as document_detail_router
from ragitect.api.v1.documents import router as documents_router
from ragitect.api.v1.llm_configs import router as llm_configs_router
from ragitect.api.v1.workspaces import router as workspaces_router

router = APIRouter(prefix="/api/v1")

# Include sub-routers
router.include_router(workspaces_router)
router.include_router(llm_configs_router)
router.include_router(documents_router)
router.include_router(
    document_detail_router
)  # Document-specific routes (/documents/{id})
router.include_router(chat_router)  # Chat streaming routes
