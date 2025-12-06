"""RAGitect FastAPI Application Entry Point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragitect.api.v1.router import router as api_v1_router
from ragitect.services.database.connection import DatabaseManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    # Startup: Initialize database connection
    db_manager = DatabaseManager.get_instance()
    await db_manager.initialize()
    yield
    # Shutdown: Close database connection
    await db_manager.close()


app = FastAPI(
    title="RAGitect API",
    description="RAG-powered document intelligence API",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://frontend:3000",  # Docker network
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint returning API status."""
    return {"message": "RAGitect API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/orchestration systems."""
    return {"status": "healthy"}


# Include API routers
app.include_router(api_v1_router)


def main():
    """Run the application using uvicorn."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
