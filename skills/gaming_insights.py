"""Gaming insights skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for analyzing gaming habits and insights.
    
    Args:
        llm: Language model instance
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for gaming insights
    """
    
    def gaming_insights(query: str = "") -> str:
        """Analyze gaming habits, preferences, and their relationship to mood and productivity."""
        try:
            # Search for gaming-related content
            gaming_keywords = [
                "game", "gaming", "played", "playing", "finished",
                "Steam", "PlayStation", "Nintendo", "Xbox",
                "RPG", "strategy", "puzzle", "action", "indie",
                "boss", "level", "quest", "achievement"
            ]
            
            search_query = " ".join(gaming_keywords) if not query else query
            docs = vectorstore.similarity_search(search_query, k=20)
            
            if not docs:
                return "No gaming-related entries found in your notes."
            
            # Sort by date
            docs_sorted = sorted(
                docs,
                key=lambda x: x.metadata.get('modified_time', ''),
                reverse=False
            )
            
            # Combine note contents
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('modified_time', 'Unknown')[:10]} - {doc.metadata.get('file_name', 'Unknown')} ---\n{doc.page_content}"
                for doc in docs_sorted
            ])
            
            # Load prompt template
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "gaming_insights.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                prompt = prompt_template.replace("{{notes}}", notes_text)
            else:
                prompt = f"""Analyze gaming habits and insights:

{notes_text}

Provide insights on games played, preferences, patterns, and relationship to mood/productivity.
"""
            
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in gaming_insights: {e}")
            return f"Error analyzing gaming habits: {str(e)}"
    
    return Tool(
        name="gaming_insights",
        func=gaming_insights,
        description=(
            "Analyze gaming habits, preferences, and patterns from your notes. "
            "Identifies games played, favorite genres, gaming frequency, and explores "
            "the relationship between gaming and mood/productivity. "
            "Useful for understanding your gaming behavior and finding healthy balance."
        )
    )
