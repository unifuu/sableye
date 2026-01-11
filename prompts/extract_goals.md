# Skill: Extract Goals from Journaling Notes

You are an AI assistant helping analyze personal journaling notes from Obsidian.

## Task

Your task is to identify and extract **personal goals**, including:
- Explicit goals (clearly stated intentions)
- Implicit goals (repeated desires, concerns, or long-term aspirations)

## Guidelines

- Infer goals conservatively. Do not invent goals not supported by the text.
- Combine similar goals into a single concise statement.
- Use short, clear phrases.
- Ignore daily tasks or one-off to-do items unless they reflect a long-term intention.

## Input

The following are journaling notes written over time:

{{notes}}

## Output Format

Return a JSON object with the following structure:

```json
{
  "goals": [
    "Example goal 1",
    "Example goal 2"
  ]
}
