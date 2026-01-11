"""Extract technical learnings skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for extracting technical learnings from development notes.
    
    Args:
        llm: Language model instance
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for extracting learnings
    """
    
    def extract_learnings(query: str = "") -> str:
        """Extract technical learnings and new concepts from development notes."""
        try:
            # Search for development and learning-related content
            dev_keywords = [
                "learned", "discovered", "figured out", "TIL", "today I learned",
                "programming", "coding", "development", "bug", "solution",
                "technology", "framework", "library", "API", "tutorial"
            ]
            
            search_query = " ".join(dev_keywords) if not query else query
            docs = vectorstore.similarity_search(search_query, k=15)
            
            if not docs:
                return "No development or learning-related entries found."
            
            # Combine note contents
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('file_name', 'Unknown')} ({doc.metadata.get('modified_time', '')[:10]}) ---\n{doc.page_content}"
                for doc in docs
            ])
            
            # Load prompt template
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "extract_learnings.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                prompt = prompt_template.replace("{{notes}}", notes_text)
            else:
                prompt = f"""Analyze these development notes and extract technical learnings:

{notes_text}

Provide a structured summary of new technologies, concepts, and problem-solving patterns learned.
"""
            
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in extract_learnings: {e}")
            return f"Error extracting learnings: {str(e)}"
    
    return Tool(
        name="extract_learnings",
        func=extract_learnings,
        description=(
            "Extract and summarize technical learnings from development and programming notes. "
            "Identifies new technologies, concepts, problem-solving patterns, and knowledge gained. "
            "Useful for reviewing what you've learned or creating a knowledge summary."
        )
    )
