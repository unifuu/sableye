# sableye/skills/extract_goals.py

from typing import List

def extract_goals_from_notes(notes: List[str], llm) -> List[str]:
    """
    Extract personal goals from Obsidian journaling notes.

    Args:
        notes: A list of journaling note contents.
        llm: LLM client used for extraction.

    Returns:
        A list of inferred personal goals.
    """
    prompt = f"""
    The following are personal journaling notes.

    Identify any explicit or implicit personal goals.

    Notes:
    {notes}
    """

    response = llm(prompt)
    return response["goals"]
