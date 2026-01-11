"""Energy tracker skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for analyzing energy levels and productivity patterns.
    
    Args:
        llm: Language model instance
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for energy tracking
    """
    
    def energy_tracker(days: str = "30") -> str:
        """Analyze energy levels and productivity patterns from journal entries."""
        try:
            # Parse days parameter
            try:
                n_days = int(days)
            except ValueError:
                n_days = 30
            
            # Get recent entries
            recent_docs = reader.read_recent_notes(n_days)
            
            if not recent_docs:
                return f"No entries found in the last {n_days} days."
            
            # Search for energy-related keywords
            energy_keywords = [
                "energy", "tired", "exhausted", "energized", "motivated",
                "focus", "focused", "productive", "unproductive", "lazy",
                "alert", "sleepy", "refreshed", "drained", "burnout"
            ]
            
            energy_query = " ".join(energy_keywords)
            energy_docs = vectorstore.similarity_search(energy_query, k=20)
            
            # Combine both sources
            all_docs = {doc.page_content: doc for doc in recent_docs + energy_docs}
            unique_docs = list(all_docs.values())
            
            # Sort by date
            unique_docs.sort(key=lambda x: x.metadata.get('modified_time', ''))
            
            # Combine note contents
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('modified_time', 'Unknown')[:10]} - {doc.metadata.get('file_name', 'Unknown')} ---\n{doc.page_content}"
                for doc in unique_docs[:25]  # Limit to avoid token overflow
            ])
            
            # Load prompt template
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "energy_tracker.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                prompt = prompt_template.replace("{{notes}}", notes_text)
            else:
                prompt = f"""Analyze energy levels and productivity patterns:

{notes_text}

Identify high/low energy periods, energy boosters/drains, and provide schedule recommendations.
"""
            
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in energy_tracker: {e}")
            return f"Error tracking energy: {str(e)}"
    
    return Tool(
        name="energy_tracker",
        func=energy_tracker,
        description=(
            "Analyze energy levels and productivity patterns from journal entries. "
            "Input should be number of days to analyze (default: 30). "
            "Identifies high/low energy periods, energy boosters and drains, and provides "
            "recommendations for optimal daily schedule."
        )
    )
