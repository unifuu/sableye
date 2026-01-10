import pytest
from datetime import datetime, timedelta
from src.reader import ObsidianReader
from langchain_core.documents import Document


class TestObsidianReader:
    """Test cases for ObsidianReader class"""
    
    def test_initialization(self, temp_vault_dir):
        """Test ObsidianReader initialization"""
        reader = ObsidianReader(
            vault_path=str(temp_vault_dir),
            chunk_size=1000,
            chunk_overlap=200
        )
        
        assert reader.vault_path == temp_vault_dir
        assert reader.chunk_size == 1000
        assert reader.chunk_overlap == 200
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with invalid vault path"""
        with pytest.raises(ValueError, match="Vault path does not exist"):
            ObsidianReader(vault_path="/nonexistent/path")
    
    def test_read_all_notes(self, temp_vault_dir):
        """Test reading all notes from vault"""
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        documents = reader.read_all_notes()
        
        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
    
    def test_read_all_notes_metadata(self, temp_vault_dir):
        """Test that documents have correct metadata"""
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        documents = reader.read_all_notes()
        
        for doc in documents:
            assert "source" in doc.metadata
            assert "file_name" in doc.metadata
            assert "modified_time" in doc.metadata
            assert "created_time" in doc.metadata
    
    def test_read_all_notes_skips_empty_files(self, temp_vault_dir):
        """Test that empty files are skipped"""
        (temp_vault_dir / "empty.md").write_text("")
        
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        documents = reader.read_all_notes()
        
        # Should only include non-empty files
        assert all(len(doc.page_content.strip()) > 0 for doc in documents)
    
    def test_read_recent_notes(self, temp_vault_dir):
        """Test reading recent notes within time window"""
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        
        # All test files should be recent
        recent_docs = reader.read_recent_notes(days=30)
        
        assert len(recent_docs) == 3
    
    def test_read_recent_notes_old_files(self, temp_vault_dir):
        """Test reading recent notes filters out old files"""
        # Create an old file
        old_file = temp_vault_dir / "old.md"
        old_file.write_text("Old content")
        
        # Set modification time to 60 days ago
        old_time = datetime.now() - timedelta(days=60)
        import os
        os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))
        
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        recent_docs = reader.read_recent_notes(days=30)
        
        # Old file should not be included
        assert not any(doc.metadata["file_name"] == "old.md" for doc in recent_docs)
    
    def test_split_documents(self, temp_vault_dir):
        """Test splitting documents into chunks"""
        # Create a larger file to test splitting
        large_file = temp_vault_dir / "large.md"
        large_content = "\n\n".join(["Paragraph " + str(i) for i in range(50)])
        large_file.write_text(large_content)
        
        reader = ObsidianReader(
            vault_path=str(temp_vault_dir),
            chunk_size=200,
            chunk_overlap=50
        )
        
        documents = reader.read_all_notes()
        splits = reader.split_documents(documents)
        
        # Large file should be split into multiple chunks
        large_doc_splits = [s for s in splits if "large.md" in s.metadata["source"]]
        assert len(large_doc_splits) > 1
    
    def test_split_documents_preserves_metadata(self, temp_vault_dir):
        """Test that split documents preserve original metadata"""
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        documents = reader.read_all_notes()
        splits = reader.split_documents(documents)
        
        for split in splits:
            assert "source" in split.metadata
            assert "file_name" in split.metadata
            assert "modified_time" in split.metadata
            assert "created_time" in split.metadata
    
    def test_read_notes_with_nested_directories(self, temp_vault_dir):
        """Test reading notes from nested directory structure"""
        reader = ObsidianReader(vault_path=str(temp_vault_dir))
        documents = reader.read_all_notes()
        
        # Should include notes from subdirectories
        sources = [doc.metadata["source"] for doc in documents]
        assert any("subdir" in source for source in sources)
