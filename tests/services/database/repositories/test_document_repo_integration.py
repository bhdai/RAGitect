"""Integration tests for DocumentRepository.

Requirements:
    - Running PostgreSQL database (ragitect_test)
    - DATABASE_URL environment variable set
"""

import pytest
import hashlib
from ragitect.services.database.repositories.document_repo import DocumentRepository
from ragitect.services.database.repositories.workspace_repo import WorkspaceRepository
from ragitect.services.database import get_session

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class TestDocumentRepositoryIntegration:
    """Integration tests for DocumentRepository"""

    async def test_create_document_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)

            workspace = await ws_repo.create("Doc Test WS")

            doc = await doc_repo.create(
                workspace_id=workspace.id,
                file_name="test.pdf",
                processed_content="content",
                embedding=[0.1] * 768,
            )

            assert doc.id is not None
            assert doc.workspace_id == workspace.id

            # Verify persistence
            fetched = await doc_repo.get_by_id(doc.id)
            assert fetched is not None
            assert fetched.file_name == "test.pdf"

    async def test_check_duplicate_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            workspace = await ws_repo.create("Dup Check WS")

            content = "duplicate content"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Create first doc
            await doc_repo.create(
                workspace_id=workspace.id,
                file_name="doc1.pdf",
                processed_content=content,
                embedding=[0.1] * 768,
            )

            # Check duplicate
            is_dup, existing = await doc_repo.check_duplicate(
                workspace.id, "doc2.pdf", content_hash
            )
            assert is_dup is True
            assert existing is not None

    async def test_chunks_operations_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            workspace = await ws_repo.create("Chunks WS")

            doc = await doc_repo.create(
                workspace_id=workspace.id,
                file_name="chunks.pdf",
                processed_content="content",
                embedding=[0.1] * 768,
            )

            # Add chunks
            chunks_data = [
                ("chunk 1", [0.1] * 768, {"idx": 1}),
                ("chunk 2", [0.2] * 768, {"idx": 2}),
            ]

            created_chunks = await doc_repo.add_chunks(doc.id, chunks_data)
            assert len(created_chunks) == 2

            # Get chunks
            fetched_chunks = await doc_repo.get_chunks(doc.id)
            assert len(fetched_chunks) == 2
            assert fetched_chunks[0].content == "chunk 1"

            # Count chunks
            count = await doc_repo.count_chunks(doc.id)
            assert count == 2

    async def test_update_operations_integration(self, clean_database):
        async with get_session() as session:
            ws_repo = WorkspaceRepository(session)
            doc_repo = DocumentRepository(session)
            workspace = await ws_repo.create("Update WS")

            doc = await doc_repo.create(
                workspace_id=workspace.id,
                file_name="update.pdf",
                processed_content="content",
                embedding=[0.0] * 768,
            )

            # Update embedding
            new_emb = [0.9] * 768
            _ = await doc_repo.update_embedding(doc.id, new_emb)

            # Update metadata
            new_meta = {"status": "processed"}
            updated_doc_meta = await doc_repo.update_metadata(doc.id, new_meta)
            assert updated_doc_meta.metadata_ == new_meta
