# AGENTS.md

This file contains guidelines for agentic coding assistants working on this repository.

## Project Overview

Sableye is an AI-powered personal knowledge assistant that analyzes Obsidian journaling notes using LangChain with OpenAI or Ollama LLMs.

## Commands

### Running the Application
```bash
python cli.py                    # Start interactive chat
python cli.py chat               # Explicit chat command
python cli.py ask "query"        # Single question mode
python cli.py validate           # Validate configuration
```

### Development
```bash
python3 -m venv venv             # Create virtual environment
source venv/bin/activate         # Activate venv (macOS/Linux)
pip install -r requirements.txt  # Install dependencies
```

### Testing
No test framework is currently set up. When adding tests, use pytest:
```bash
pip install pytest pytest-cov
pytest                          # Run all tests
pytest tests/test_specific.py    # Run single test file
pytest -k "test_function"       # Run specific test
pytest -x                       # Stop on first failure
pytest --cov=src                # Generate coverage report
```

### Linting/Formatting (Recommended)
```bash
pip install ruff black isort mypy
ruff check src/                 # Lint code
ruff check --fix src/           # Auto-fix linting issues
black src/                      # Format code
isort src/                      # Sort imports
mypy src/                       # Type checking
```

## Code Style Guidelines

### Imports
- Standard library imports first, then third-party, then local imports
- Use `from .` for same-package imports (e.g., `from .reader import ObsidianReader`)
- Group imports with blank lines between each group
- Avoid wildcard imports (`from module import *`)

### Naming Conventions
- Classes: PascalCase (e.g., `SableyeAgent`, `ModelConfig`)
- Functions/Methods: snake_case (e.g., `load_notes`, `_initialize_models`)
- Constants: UPPER_SNAKE_CASE
- Private methods: prefix with underscore
- Variables: snake_case

### Type Hints
- Use type hints for all function signatures and class attributes
- Import from `typing` module: `Optional`, `List`, `Dict`, `Any`, `Callable`
- Use dataclasses for configuration classes with type hints
- Use `Optional[T]` for nullable types (e.g., `days: Optional[int] = None`)

### Docstrings
- Use triple-quoted strings for all class and method docstrings
- Keep docstrings concise and descriptive
- Format: `"""Short description. Optionally longer description."""`

### Logging
- Use `logger = logging.getLogger(__name__)` at module level
- Log at appropriate levels: `logger.info()`, `logger.error()`, `logger.debug()`
- Include context in error logs with `exc_info=True`

### Error Handling
- Use try/except blocks for operations that may fail
- Log errors with context before raising or returning
- Use descriptive error messages in exceptions
- Return user-friendly error strings from tool functions

### Configuration
- Use `@dataclass` for configuration classes with defaults
- Support both YAML files and environment variables
- Use `config.local.yaml` for local overrides (gitignored)
- Validate configuration on load with `validate()` method

### File Structure
- Keep CLI in root `cli.py`
- Core logic in `src/` directory
- Use `__init__.py` for package exports
- Configuration files in root: `config.yaml`, `config.local.yaml`

### Git Ignore
- Never commit `.env*` files, `config.local.yaml`, or secrets
- Ignore `venv/`, `__pycache__/`, `*.log`
- Ignore Obsidian internal files: `.obsidian/`

## LangChain Integration
- Use `ChatOpenAI` or `ChatOllama` for LLMs
- Use `OpenAIEmbeddings` or `OllamaEmbeddings` for vectorization
- Use `FAISS` for vector storage (local, no external DB)
- Use `Tool` class for agent tools with descriptive docstrings
- Use `ConversationBufferMemory` for chat history
- Use `create_tool_calling_agent` and `AgentExecutor` for agent execution
- Set `handle_parsing_errors=True` in `AgentExecutor` for graceful error recovery
- Use `MessagesPlaceholder` for dynamic prompt variables like `chat_history`

## CLI Development (Click)
- Use `@click.group()` for command groups
- Use `@click.command()` for subcommands
- Use `@click.option()` for flags and parameters
- Use Rich for beautiful terminal output: `Console()`, `Panel()`, `Markdown()`

## Environment Variables
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI)
- `OBSIDIAN_VAULT_PATH`: Path to Obsidian vault
- `MODEL_TYPE`: "openai" or "ollama"
- `MODEL_NAME`: Model name (e.g., "gpt-4", "qwen3:4b-instruct")

## Agent Tool Conventions
- Tools should return user-friendly string responses
- Wrap tool logic in try/except blocks and log errors
- Include descriptive docstrings explaining when and how to use the tool
- Use `similarity_search` on vectorstores for searching content
- Limit results returned from tools (e.g., top 5-10 results)

## Document Processing
- Use `RecursiveCharacterTextSplitter` for chunking documents
- Metadata should include: `source`, `file_name`, `modified_time`, `created_time`
- Store timestamps in ISO format using `.isoformat()`
- Use `Path.rglob("*.md")` to recursively find markdown files
- Skip empty documents when processing vault contents
