"""Track progress skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for tracking progress on goals and habits.
    
    Args:
        llm: Language model instance
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for tracking progress
    """
    
    def track_progress(goal: str) -> str:
        """Track progress on a specific goal or habit over time."""
        try:
            if not goal:
                return "Please specify a goal or habit to track (e.g., 'learning Rust', 'daily exercise')."
            
            # Search for mentions of the goal
            docs = vectorstore.similarity_search(goal, k=20)
            
            if not docs:
                return f"No entries found mentioning '{goal}'."
            
            # Sort by date
            docs_sorted = sorted(
                docs,
                key=lambda x: x.metadata.get('modified_time', ''),
                reverse=False  # Chronological order
            )
            
            # Combine note contents with dates
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('modified_time', 'Unknown date')[:10]} - {doc.metadata.get('file_name', 'Unknown')} ---\n{doc.page_content}"
                for doc in docs_sorted
            ])
            
            # Load prompt template
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "track_progress.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                prompt = prompt_template.replace("{{goal}}", goal).replace("{{notes}}", notes_text)
            else:
                prompt = f"""Track progress for: {goal}

Entries:
{notes_text}

Analyze the progress, identify patterns, and provide recommendations.
"""
            
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in track_progress: {e}")
            return f"Error tracking progress: {str(e)}"
    
    return Tool(
        name="track_progress",
        func=track_progress,
        description=(
            "Track progress on a specific goal, habit, or project over time. "
            "Input should be the goal/habit name (e.g., 'learning Python', 'daily meditation'). "
            "Analyzes mentions over time, identifies patterns, and provides progress assessment."
        )
    )
