"""Ask past self skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool
from pathlib import Path

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a tool for querying past notes to answer questions.
    
    Args:
        llm: Language model instance
        vectorstore: Vector store for searching notes
        reader: Reader instance for accessing notes
        
    Returns:
        LangChain Tool for asking past self
    """
    
    def ask_past_self(question: str) -> str:
        """Query historical notes to find answers from past experiences."""
        try:
            if not question:
                return "Please provide a question to search for in your past notes."
            
            # Search for relevant past entries
            docs = vectorstore.similarity_search(question, k=12)
            
            if not docs:
                return f"No relevant entries found for: {question}"
            
            # Combine note contents with metadata
            notes_text = "\n\n".join([
                f"--- {doc.metadata.get('modified_time', 'Unknown')[:10]} - {doc.metadata.get('file_name', 'Unknown')} ---\n{doc.page_content}"
                for doc in docs
            ])
            
            # Load prompt template
            prompts_dir = Path(__file__).parent.parent / "prompts"
            prompt_file = prompts_dir / "ask_past_self.md"
            
            if prompt_file.exists():
                prompt_template = prompt_file.read_text(encoding='utf-8')
                prompt = prompt_template.replace("{{question}}", question).replace("{{notes}}", notes_text)
            else:
                prompt = f"""Question: {question}

Past entries:
{notes_text}

Based on these past notes, provide an answer with supporting evidence and dates.
"""
            
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ask_past_self: {e}")
            return f"Error querying past notes: {str(e)}"
    
    return Tool(
        name="ask_past_self",
        func=ask_past_self,
        description=(
            "Query your historical notes to find answers from your past experiences and thoughts. "
            "Input should be a question (e.g., 'How did I solve the authentication bug last time?', "
            "'What did I think about React hooks when I first learned them?'). "
            "Searches past entries and synthesizes an answer with supporting evidence."
        )
    )
