import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ObsidianReader:
    """Read and process Obsidian markdown files"""
    
    def __init__(self, vault_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.vault_path = Path(vault_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def read_all_notes(self) -> List[Document]:
        """Read all markdown files from Obsidian vault"""
        documents = []
        md_files = list(self.vault_path.rglob("*.md"))
        
        logger.info(f"Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Extract metadata
                metadata = {
                    "source": str(md_file.relative_to(self.vault_path)),
                    "file_name": md_file.name,
                    "modified_time": datetime.fromtimestamp(
                        md_file.stat().st_mtime
                    ).isoformat(),
                    "created_time": datetime.fromtimestamp(
                        md_file.stat().st_ctime
                    ).isoformat()
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                
            except Exception as e:
                logger.error(f"Error reading {md_file}: {e}")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def read_recent_notes(self, days: int = 30) -> List[Document]:
        """Read notes modified in the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        all_docs = self.read_all_notes()
        
        recent_docs = [
            doc for doc in all_docs
            if datetime.fromisoformat(doc.metadata["modified_time"]) > cutoff_date
        ]
        
        logger.info(f"Found {len(recent_docs)} documents from last {days} days")
        return recent_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        splits = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(splits)} chunks")
        return splits