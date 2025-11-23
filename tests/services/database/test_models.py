"""Tests for database ORM models"""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, inspect, text
from sqlalchemy.exc import IntegrityError

from ragitect.services.database import get_session
from ragitect.services.database.models import Base, Document, DocumentChunk, Workspace


class TestBaseModel:
    """Test Base declarative base"""

    def test_base_is_abstract(self):
        """Test that Base is marked as abstract"""
        assert Base.__abstract__ is True

    def test_base_has_no_tablename(self):
        """Test that Base does not define a table"""
        assert not hasattr(Base, "__tablename__")


class TestWorkspaceModel:
    """Test Workspace ORM model"""

    def test_workspace_tablename(self):
        """Test workspace table name is correct"""
        assert Workspace.__tablename__ == "workspaces"

    def test_workspace_fields_exist(self):
        """Test all expected fields exist on Workspace model"""
        inspector = inspect(Workspace)
        column_names = [col.name for col in inspector.columns]

        expected_fields = ["id", "name", "description", "created_at", "updated_at"]
        for field in expected_fields:
            assert field in column_names, f"Field {field} not found in Workspace model"

    def test_workspace_has_documents_relationship(self):
        """Test Workspace has documents relationship"""
        assert hasattr(Workspace, "documents")
        # Check relationship is configured
        relationships = inspect(Workspace).relationships
        assert "documents" in relationships.keys()

    def test_workspace_name_has_unique_constraint(self):
        """Test name field has unique constraint"""
        inspector = inspect(Workspace)
        name_col = inspector.columns["name"]
        assert name_col.unique is True

    def test_workspace_has_check_constraint_for_name(self):
        """Test workspace has check constraint for non-empty name"""
        constraints = Workspace.__table_args__[0]
        assert isinstance(constraints, CheckConstraint)
        assert "length(name) > 0" in str(constraints.sqltext)

    def test_workspace_repr(self):
        """Test Workspace __repr__ method"""
        workspace = Workspace()
        workspace.id = uuid.uuid4()
        workspace.name = "Test Workspace"

        repr_str = repr(workspace)

        assert "Workspace" in repr_str
        assert str(workspace.id) in repr_str
        assert "Test Workspace" in repr_str

    def test_workspace_id_is_uuid(self):
        """Test workspace id field is UUID type"""
        inspector = inspect(Workspace)
        id_col = inspector.columns["id"]
        assert "UUID" in str(id_col.type)

    def test_workspace_created_at_has_server_default(self):
        """Test created_at has server default"""
        inspector = inspect(Workspace)
        created_at_col = inspector.columns["created_at"]
        assert created_at_col.server_default is not None

    def test_workspace_updated_at_has_onupdate(self):
        """Test updated_at has onupdate configured"""
        inspector = inspect(Workspace)
        updated_at_col = inspector.columns["updated_at"]
        assert updated_at_col.server_default is not None


class TestDocumentModel:
    """Test Document ORM model"""

    def test_document_tablename(self):
        """Test document table name is correct"""
        assert Document.__tablename__ == "documents"

    def test_document_fields_exist(self):
        """Test all expected fields exist on Document model"""
        inspector = inspect(Document)
        column_names = [col.name for col in inspector.columns]

        expected_fields = [
            "id",
            "workspace_id",
            "file_name",
            "file_type",
            "content_hash",
            "unique_identifier_hash",
            "processed_content",
            "summary",
            "embedding",
            "processed_at",
            "metadata",
        ]
        for field in expected_fields:
            assert field in column_names, f"Field {field} not found in Document model"

    def test_document_has_workspace_relationship(self):
        """Test Document has workspace relationship"""
        assert hasattr(Document, "workspace")
        relationships = inspect(Document).relationships
        assert "workspace" in relationships.keys()

    def test_document_has_chunks_relationship(self):
        """Test Document has chunks relationship"""
        assert hasattr(Document, "chunks")
        relationships = inspect(Document).relationships
        assert "chunks" in relationships.keys()

    def test_document_workspace_id_is_foreign_key(self):
        """Test workspace_id is a foreign key"""
        inspector = inspect(Document)
        workspace_id_col = inspector.columns["workspace_id"]
        assert len(list(workspace_id_col.foreign_keys)) > 0

    def test_document_has_unique_constraint(self):
        """Test document has unique constraint on workspace_id and content_hash"""
        table = Document.__table__
        unique_constraints = [
            c for c in table.constraints if isinstance(c, UniqueConstraint)
        ]

        # Find the specific constraint
        workspace_content_constraint = None
        for constraint in unique_constraints:
            col_names = [col.name for col in constraint.columns]
            if "workspace_id" in col_names and "content_hash" in col_names:
                workspace_content_constraint = constraint
                break

        assert workspace_content_constraint is not None, (
            "Unique constraint on workspace_id and content_hash not found"
        )

    def test_document_unique_identifier_hash_is_unique(self):
        """Test unique_identifier_hash has unique constraint"""
        inspector = inspect(Document)
        unique_id_hash_col = inspector.columns["unique_identifier_hash"]
        assert unique_id_hash_col.unique is True

    def test_document_embedding_is_vector_type(self):
        """Test embedding field is Vector type"""
        inspector = inspect(Document)
        embedding_col = inspector.columns["embedding"]
        assert "vector" in str(embedding_col.type).lower()

    def test_document_metadata_is_jsonb(self):
        """Test metadata_ field is JSONB type"""
        inspector = inspect(Document)
        metadata_col = inspector.columns["metadata_"]
        assert "JSONB" in str(metadata_col.type) or "JSON" in str(metadata_col.type)

    def test_document_repr(self):
        """Test Document __repr__ method"""
        document = Document()
        document.id = uuid.uuid4()
        document.file_name = "test.pdf"
        document.workspace_id = uuid.uuid4()

        repr_str = repr(document)

        assert "Document" in repr_str
        assert str(document.id) in repr_str
        assert "test.pdf" in repr_str
        assert str(document.workspace_id) in repr_str


class TestDocumentChunkModel:
    """Test DocumentChunk ORM model"""

    def test_document_chunk_tablename(self):
        """Test document_chunk table name is correct"""
        assert DocumentChunk.__tablename__ == "document_chunks"

    def test_document_chunk_fields_exist(self):
        """Test all expected fields exist on DocumentChunk model"""
        inspector = inspect(DocumentChunk)
        column_names = [col.name for col in inspector.columns]

        expected_fields = [
            "id",
            "document_id",
            "workspace_id",
            "chunk_index",
            "content",
            "embedding",
            "metadata",
            "created_at",
        ]
        for field in expected_fields:
            assert field in column_names, (
                f"Field {field} not found in DocumentChunk model"
            )

    def test_document_chunk_has_document_relationship(self):
        """Test DocumentChunk has document relationship"""
        assert hasattr(DocumentChunk, "document")
        relationships = inspect(DocumentChunk).relationships
        assert "document" in relationships.keys()

    def test_document_chunk_document_id_is_foreign_key(self):
        """Test document_id is a foreign key"""
        inspector = inspect(DocumentChunk)
        document_id_col = inspector.columns["document_id"]
        assert len(list(document_id_col.foreign_keys)) > 0

    def test_document_chunk_workspace_id_is_foreign_key(self):
        """Test workspace_id is a foreign key"""
        inspector = inspect(DocumentChunk)
        workspace_id_col = inspector.columns["workspace_id"]
        assert len(list(workspace_id_col.foreign_keys)) > 0

    def test_document_chunk_has_unique_constraint(self):
        """Test chunk has unique constraint on document_id and chunk_index"""
        table = DocumentChunk.__table__
        unique_constraints = [
            c for c in table.constraints if isinstance(c, UniqueConstraint)
        ]

        # Find the specific constraint
        doc_chunk_constraint = None
        for constraint in unique_constraints:
            col_names = [col.name for col in constraint.columns]
            if "document_id" in col_names and "chunk_index" in col_names:
                doc_chunk_constraint = constraint
                break

        assert doc_chunk_constraint is not None, (
            "Unique constraint on document_id and chunk_index not found"
        )

    def test_document_chunk_has_check_constraint_for_index(self):
        """Test chunk has check constraint for non-negative index"""
        table = DocumentChunk.__table__
        check_constraints = [
            c for c in table.constraints if isinstance(c, CheckConstraint)
        ]

        index_constraint = None
        for constraint in check_constraints:
            if "chunk_index >= 0" in str(constraint.sqltext):
                index_constraint = constraint
                break

        assert index_constraint is not None, (
            "Check constraint for chunk_index >= 0 not found"
        )

    def test_document_chunk_has_check_constraint_for_content(self):
        """Test chunk has check constraint for non-empty content"""
        table = DocumentChunk.__table__
        check_constraints = [
            c for c in table.constraints if isinstance(c, CheckConstraint)
        ]

        content_constraint = None
        for constraint in check_constraints:
            if "length(content) > 0" in str(constraint.sqltext):
                content_constraint = constraint
                break

        assert content_constraint is not None, (
            "Check constraint for non-empty content not found"
        )

    def test_document_chunk_embedding_is_vector_type(self):
        """Test embedding field is Vector type"""
        inspector = inspect(DocumentChunk)
        embedding_col = inspector.columns["embedding"]
        assert "vector" in str(embedding_col.type).lower()

    def test_document_chunk_metadata_is_jsonb(self):
        """Test metadata_ field is JSONB type"""
        inspector = inspect(DocumentChunk)
        metadata_col = inspector.columns["metadata_"]
        assert "JSONB" in str(metadata_col.type) or "JSON" in str(metadata_col.type)

    def test_document_chunk_repr(self):
        """Test DocumentChunk __repr__ method"""
        chunk = DocumentChunk()
        chunk.id = uuid.uuid4()
        chunk.document_id = uuid.uuid4()
        chunk.chunk_index = 5
        chunk.content = "This is a test content for chunk representation"

        repr_str = repr(chunk)

        assert "DocumentChunk" in repr_str
        assert str(chunk.id) in repr_str
        assert str(chunk.document_id) in repr_str
        assert str(chunk.chunk_index) in repr_str
        assert "This is a test content" in repr_str

    def test_document_chunk_repr_truncates_long_content(self):
        """Test DocumentChunk __repr__ truncates long content"""
        chunk = DocumentChunk()
        chunk.id = uuid.uuid4()
        chunk.document_id = uuid.uuid4()
        chunk.chunk_index = 0
        chunk.content = "x" * 100  # 100 characters

        repr_str = repr(chunk)

        # Should truncate to 50 chars + "..."
        assert "..." in repr_str
        assert len(chunk.content) > 50


class TestModelRelationships:
    """Test relationships between models"""

    def test_workspace_documents_cascade_delete(self):
        """Test workspace documents relationship has cascade delete"""
        relationships = inspect(Workspace).relationships
        documents_rel = relationships["documents"]

        # Check cascade options
        assert "delete-orphan" in documents_rel.cascade

    def test_document_chunks_cascade_delete(self):
        """Test document chunks relationship has cascade delete"""
        relationships = inspect(Document).relationships
        chunks_rel = relationships["chunks"]

        # Check cascade options
        assert "delete-orphan" in chunks_rel.cascade

    def test_workspace_documents_lazy_selectin(self):
        """Test workspace documents relationship uses selectin loading"""
        relationships = inspect(Workspace).relationships
        documents_rel = relationships["documents"]

        assert str(documents_rel.lazy) == "selectin"

    def test_document_chunks_lazy_selectin(self):
        """Test document chunks relationship uses selectin loading"""
        relationships = inspect(Document).relationships
        chunks_rel = relationships["chunks"]

        assert str(chunks_rel.lazy) == "selectin"


class TestWorkspaceModelIntegration:
    """Integration tests for Workspace model (requires database)"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_workspace(self, clean_db_manager):
        """Test creating a workspace in database"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create tables
        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            async with get_session() as session:
                workspace = Workspace(
                    name="Test Workspace", description="A test workspace"
                )
                session.add(workspace)
                await session.flush()

                assert workspace.id is not None
                assert isinstance(workspace.id, uuid.UUID)
                assert workspace.name == "Test Workspace"
                assert workspace.description == "A test workspace"
                assert isinstance(workspace.created_at, datetime)
                assert isinstance(workspace.updated_at, datetime)

        finally:
            # Cleanup
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workspace_name_unique_constraint(self, clean_db_manager):
        """Test workspace name unique constraint is enforced"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create first workspace
            async with get_session() as session:
                workspace1 = Workspace(name="Duplicate Name")
                session.add(workspace1)

            # Try to create second workspace with same name
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    workspace2 = Workspace(name="Duplicate Name")
                    session.add(workspace2)
                    await session.flush()

            assert (
                "unique" in str(exc_info.value).lower()
                or "duplicate" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_workspace_name_not_empty_constraint(self, clean_db_manager):
        """Test workspace name cannot be empty"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    workspace = Workspace(name="")
                    session.add(workspace)
                    await session.flush()

            assert (
                "check" in str(exc_info.value).lower()
                or "constraint" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()


class TestDocumentModelIntegration:
    """Integration tests for Document model (requires database)"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_document(self, clean_db_manager):
        """Test creating a document in database"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace first
            async with get_session() as session:
                workspace = Workspace(name="Doc Test Workspace")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

            # Create document
            async with get_session() as session:
                test_embedding = [0.1] * 768  # 768-dimensional vector
                document = Document(
                    workspace_id=workspace_id,
                    file_name="test.pdf",
                    file_type="pdf",
                    content_hash="abc123",
                    unique_identifier_hash="unique123",
                    processed_content="Test content",
                    summary="Test summary",
                    embedding=test_embedding,
                    metadata_={"page_count": 10},
                )
                session.add(document)
                await session.flush()

                assert document.id is not None
                assert isinstance(document.id, uuid.UUID)
                assert document.workspace_id == workspace_id
                assert document.file_name == "test.pdf"
                assert document.file_type == "pdf"
                assert document.metadata_ == {"page_count": 10}

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_workspace_content_hash_unique(self, clean_db_manager):
        """Test document workspace_id + content_hash unique constraint"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace
            async with get_session() as session:
                workspace = Workspace(name="Unique Test Workspace")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

            # Create first document
            async with get_session() as session:
                test_embedding = [0.1] * 768
                doc1 = Document(
                    workspace_id=workspace_id,
                    file_name="test1.pdf",
                    file_type="pdf",
                    content_hash="same_hash",
                    unique_identifier_hash="unique1",
                    embedding=test_embedding,
                )
                session.add(doc1)

            # Try to create duplicate (same workspace + content_hash)
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    doc2 = Document(
                        workspace_id=workspace_id,
                        file_name="test2.pdf",
                        file_type="pdf",
                        content_hash="same_hash",  # Same hash
                        unique_identifier_hash="unique2",
                        embedding=test_embedding,
                    )
                    session.add(doc2)
                    await session.flush()

            assert (
                "unique" in str(exc_info.value).lower()
                or "duplicate" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_cascade_delete_from_workspace(self, clean_db_manager):
        """Test that deleting workspace cascades to documents"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace and document
            async with get_session() as session:
                workspace = Workspace(name="Cascade Test Workspace")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                test_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="cascade_test.pdf",
                    content_hash="cascade_hash",
                    unique_identifier_hash="cascade_unique",
                    embedding=test_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

            # Delete workspace
            async with get_session() as session:
                workspace = await session.get(Workspace, workspace_id)
                await session.delete(workspace)

            # Verify document is also deleted
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM documents WHERE id = :id"),
                    {"id": str(document_id)},
                )
                count = result.scalar()
                assert count == 0, "Document should be cascade deleted"

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()


class TestDocumentChunkModelIntegration:
    """Integration tests for DocumentChunk model (requires database)"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_create_document_chunk(self, clean_db_manager):
        """Test creating a document chunk in database"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace and document
            async with get_session() as session:
                workspace = Workspace(name="Chunk Test Workspace")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                doc_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="chunk_test.pdf",
                    content_hash="chunk_hash",
                    unique_identifier_hash="chunk_unique",
                    embedding=doc_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

            # Create chunk
            async with get_session() as session:
                chunk_embedding = [0.2] * 768
                chunk = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=0,
                    content="This is chunk content",
                    embedding=chunk_embedding,
                    metadata_={"page": 1},
                )
                session.add(chunk)
                await session.flush()

                assert chunk.id is not None
                assert isinstance(chunk.id, uuid.UUID)
                assert chunk.document_id == document_id
                assert chunk.workspace_id == workspace_id
                assert chunk.chunk_index == 0
                assert chunk.content == "This is chunk content"
                assert chunk.metadata_ == {"page": 1}

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_chunk_unique_constraint(self, clean_db_manager):
        """Test document_id + chunk_index unique constraint"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace and document
            async with get_session() as session:
                workspace = Workspace(name="Unique Chunk Test")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                doc_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="unique_chunk.pdf",
                    content_hash="unique_chunk_hash",
                    unique_identifier_hash="unique_chunk_id",
                    embedding=doc_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

            # Create first chunk
            async with get_session() as session:
                chunk_embedding = [0.2] * 768
                chunk1 = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=0,
                    content="First chunk",
                    embedding=chunk_embedding,
                )
                session.add(chunk1)

            # Try to create duplicate chunk with same index
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    chunk2 = DocumentChunk(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        chunk_index=0,  # Same index
                        content="Second chunk",
                        embedding=chunk_embedding,
                    )
                    session.add(chunk2)
                    await session.flush()

            assert (
                "unique" in str(exc_info.value).lower()
                or "duplicate" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_chunk_negative_index_constraint(self, clean_db_manager):
        """Test chunk_index cannot be negative"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace and document
            async with get_session() as session:
                workspace = Workspace(name="Negative Index Test")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                doc_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="negative_test.pdf",
                    content_hash="negative_hash",
                    unique_identifier_hash="negative_unique",
                    embedding=doc_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

            # Try to create chunk with negative index
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    chunk_embedding = [0.2] * 768
                    chunk = DocumentChunk(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        chunk_index=-1,  # Negative index
                        content="Negative chunk",
                        embedding=chunk_embedding,
                    )
                    session.add(chunk)
                    await session.flush()

            assert (
                "check" in str(exc_info.value).lower()
                or "constraint" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_chunk_empty_content_constraint(self, clean_db_manager):
        """Test chunk content cannot be empty"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace and document
            async with get_session() as session:
                workspace = Workspace(name="Empty Content Test")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                doc_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="empty_content.pdf",
                    content_hash="empty_hash",
                    unique_identifier_hash="empty_unique",
                    embedding=doc_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

            # Try to create chunk with empty content
            with pytest.raises(IntegrityError) as exc_info:
                async with get_session() as session:
                    chunk_embedding = [0.2] * 768
                    chunk = DocumentChunk(
                        document_id=document_id,
                        workspace_id=workspace_id,
                        chunk_index=0,
                        content="",  # Empty content
                        embedding=chunk_embedding,
                    )
                    session.add(chunk)
                    await session.flush()

            assert (
                "check" in str(exc_info.value).lower()
                or "constraint" in str(exc_info.value).lower()
            )

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_chunk_cascade_delete(self, clean_db_manager):
        """Test that deleting document cascades to chunks"""
        import os

        if not os.getenv("DATABASE_URL"):
            pytest.skip("DATABASE_URL not set - skipping integration test")

        if not clean_db_manager._engine:
            await clean_db_manager.initialize()

        # Enable pgvector extension
        async with get_session() as session:
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        from ragitect.services.database.connection import create_table

        await create_table()

        try:
            # Create workspace, document, and chunk
            async with get_session() as session:
                workspace = Workspace(name="Cascade Chunk Test")
                session.add(workspace)
                await session.flush()
                workspace_id = workspace.id

                doc_embedding = [0.1] * 768
                document = Document(
                    workspace_id=workspace_id,
                    file_name="cascade_chunk.pdf",
                    content_hash="cascade_chunk_hash",
                    unique_identifier_hash="cascade_chunk_unique",
                    embedding=doc_embedding,
                )
                session.add(document)
                await session.flush()
                document_id = document.id

                chunk_embedding = [0.2] * 768
                chunk = DocumentChunk(
                    document_id=document_id,
                    workspace_id=workspace_id,
                    chunk_index=0,
                    content="Cascade chunk",
                    embedding=chunk_embedding,
                )
                session.add(chunk)
                await session.flush()
                chunk_id = chunk.id

            # Delete document
            async with get_session() as session:
                document = await session.get(Document, document_id)
                await session.delete(document)

            # Verify chunk is also deleted
            async with get_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM document_chunks WHERE id = :id"),
                    {"id": str(chunk_id)},
                )
                count = result.scalar()
                assert count == 0, "Chunk should be cascade deleted"

        finally:
            from ragitect.services.database.connection import drop_table

            await drop_table()
