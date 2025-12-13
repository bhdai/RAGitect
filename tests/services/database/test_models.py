"""Tests for database ORM models"""

import uuid
import pytest
from sqlalchemy import CheckConstraint, UniqueConstraint, inspect
from ragitect.services.database.models import Base, Document, DocumentChunk, Workspace

pytestmark = [pytest.mark.asyncio]


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
