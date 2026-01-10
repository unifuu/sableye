import pytest
import tempfile
from pathlib import Path
from src.config import Config, ModelConfig, VaultConfig, AgentConfig


@pytest.fixture
def temp_vault_dir():
    """Create a temporary directory for testing vault operations"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)
        
        # Create some test markdown files
        (vault_path / "test1.md").write_text("# Test Note 1\n\nThis is a test note.")
        (vault_path / "test2.md").write_text("# Test Note 2\n\nAnother test note.")
        
        # Create subdirectory with note
        subdir = vault_path / "subdir"
        subdir.mkdir()
        (subdir / "test3.md").write_text("# Nested Note\n\nNested content.")
        
        yield vault_path


@pytest.fixture
def sample_config():
    """Create a sample configuration"""
    return Config(config_path=None)


@pytest.fixture
def mock_config_dict():
    """Mock configuration dictionary"""
    return {
        "model": {
            "type": "ollama",
            "name": "qwen3:4b-instruct",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "vault": {
            "path": "/tmp/test_vault",
            "load_days": 30,
            "chunk_size": 300,
            "chunk_overlap": 50
        },
        "agent": {
            "verbose": False,
            "max_iterations": 5,
            "search_results_limit": 5
        }
    }
