# Skills Development Guide

This directory contains modular skills for the Sableye agent. Each skill is a Python module that exports a LangChain `Tool`.

## Creating a New Skill

### 1. Skill Structure

Each skill module must export a `create_tool()` function with this signature:

```python
def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """
    Create a LangChain Tool for this skill.

    Args:
        llm: Language model instance
        vectorstore: FAISS vector store for searching notes
        reader: ObsidianReader instance for accessing notes

    Returns:
        LangChain Tool object
    """
```

### 2. Example Skill Template

```python
"""My custom skill for Sableye agent."""

import logging
from typing import Any
from langchain_core.tools import Tool

logger = logging.getLogger(__name__)


def create_tool(llm: Any, vectorstore: Any, reader: Any) -> Tool:
    """Create the skill tool."""

    def my_skill_function(query: str = "") -> str:
        """The actual skill logic."""
        try:
            # Your skill implementation here
            # You can use:
            # - llm.invoke(prompt) to call the LLM
            # - vectorstore.similarity_search(query, k=5) to search notes
            # - reader.read_recent_notes(days) to get recent notes

            return "Skill result"

        except Exception as e:
            logger.error(f"Error in my_skill: {e}")
            return f"Error: {str(e)}"

    return Tool(
        name="my_skill",  # Unique name (snake_case)
        func=my_skill_function,
        description=(
            "Clear description of what this skill does. "
            "The agent uses this to decide when to call the skill."
        )
    )
```

### 3. Using Prompts

You can load prompts from the `prompts/` directory:

```python
from pathlib import Path

prompts_dir = Path(__file__).parent.parent / "prompts"
prompt_file = prompts_dir / "my_prompt.md"

if prompt_file.exists():
    prompt_template = prompt_file.read_text(encoding='utf-8')
    # Use {{placeholder}} in your .md file
    prompt = prompt_template.replace("{{placeholder}}", actual_value)
```

### 4. Skill Naming Conventions

- **File name**: `my_skill.py` (snake_case)
- **Tool name**: `"my_skill"` (matches file name)
- **Function name**: `create_tool()` (required)

### 5. Testing Your Skill

1. Place your skill file in `skills/` directory
2. Restart the agent
3. The skill will be automatically loaded
4. Check logs for: `"Loaded skill: my_skill"`

## Available Skills

- **extract_goals**: Intelligently extract personal goals from journal entries using LLM analysis
- **extract_learnings**: Extract technical learnings, new concepts, and knowledge from development notes
- **track_progress**: Track progress on specific goals or habits over time with timeline analysis
- **ask_past_self**: Query your historical notes to find answers from past experiences
- **energy_tracker**: Analyze energy levels and productivity patterns to optimize your schedule
- **gaming_insights**: Analyze gaming habits, preferences, and their relationship to mood/productivity

## Best Practices

1. **Error Handling**: Always wrap your skill logic in try-except
2. **Logging**: Use `logger.error()` for errors, `logger.info()` for important events
3. **Return Strings**: Tools should return human-readable strings
4. **Clear Descriptions**: Write clear tool descriptions so the agent knows when to use them
5. **Limit Results**: When searching notes, limit results to avoid overwhelming the LLM

## Troubleshooting

- **Skill not loading**: Check that your file doesn't start with `_` and has `create_tool()` function
- **Import errors**: Ensure all dependencies are installed
- **Tool not being called**: Improve the tool description to make it clearer when to use it
