"""Extract goals skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for extracting personal goals from journal notes.
    
    Args:
        llm: Language model instance for goal extraction
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for goal extraction
    """
    
    def extract_goals(query: str = "") -> str:
        """Extract personal goals from journal entries using LLM analysis."""
        try:
            # Search for goal-related content
            goal_keywords = [
                "goal", "objective", "want to", "plan to",
                "aspire", "achieve", "target", "aim",
                "working on", "working towards"
            ]
            
            goal_query = " ".join(goal_keywords)
            docs = vectorstore.similarity_search(goal_query, k=10)
            
            if not docs:
                return "No goal-related entries found in the notes."
            
            # Combine note contents
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('file_name', 'Unknown')} ---\n{doc.page_content}"
                for doc in docs
            ])
            
            # Load the extract_goals prompt
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "extract_goals.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                # Replace {{notes}} placeholder with actual notes
                prompt = prompt_template.replace("{{notes}}", notes_text)
            else:
                # Fallback prompt if file not found
                prompt = f"""Analyze the following personal journaling notes and extract personal goals.

Guidelines:
- Identify explicit goals (clearly stated intentions)
- Identify implicit goals (repeated desires, concerns, or long-term aspirations)
- Combine similar goals into concise statements
- Ignore daily tasks unless they reflect long-term intentions

Notes:
{notes_text}

Please list the identified goals in a clear, concise format.
"""
            
            # Use LLM to extract goals
            response = llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in extract_goals: {e}")
            return f"Error extracting goals: {str(e)}"
    
    return Tool(
        name="extract_goals",
        func=extract_goals,
        description=(
            "Extract and analyze personal goals from journal entries using AI. "
            "This tool intelligently identifies both explicit and implicit goals, "
            "aspirations, and long-term objectives mentioned in the notes. "
            "Use this when you need a comprehensive understanding of the user's goals."
        )
    )
