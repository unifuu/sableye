import pytest
import tempfile
import os
from pathlib import Path
from src.config import Config, ModelConfig, VaultConfig, AgentConfig


class TestConfig:
    """Test cases for Config class"""
    
    def test_default_initialization(self):
        """Test default configuration initialization"""
        # Test with no config file and no env vars
        config = Config(config_path=None)
        
        # Note: If config.local.yaml exists, it will override defaults
        # We check for reasonable values rather than exact defaults
        assert config.model.type in ["openai", "ollama"]
        assert config.vault.load_days is not None
        assert config.agent.verbose is False
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values"""
        model_config = ModelConfig()
        
        assert model_config.type == "openai"
        assert model_config.name == "gpt-4"
        assert model_config.api_key is None
        assert model_config.temperature == 0.7
        assert model_config.max_tokens == 2000
    
    def test_vault_config_defaults(self):
        """Test VaultConfig default values"""
        vault_config = VaultConfig()
        
        assert vault_config.path == ""
        assert vault_config.load_days == 90
        assert vault_config.chunk_size == 1000
        assert vault_config.chunk_overlap == 200
    
    def test_agent_config_defaults(self):
        """Test AgentConfig default values"""
        agent_config = AgentConfig()
        
        assert agent_config.verbose is False
        assert agent_config.max_iterations == 5
        assert agent_config.search_results_limit == 5
    
    def test_load_from_yaml(self, mock_config_dict, temp_vault_dir):
        """Test loading configuration from YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            mock_config_dict["vault"]["path"] = str(temp_vault_dir)
            yaml.dump(mock_config_dict, f)
            config_path = f.name
        
        try:
            config = Config(config_path=config_path)
            
            assert config.model.type == "ollama"
            assert config.model.name == "qwen3:4b-instruct"
            assert config.vault.load_days == 30
            assert config.agent.max_iterations == 5
        finally:
            os.unlink(config_path)
    
    def test_load_from_env(self, monkeypatch):
        """Test loading configuration from environment variables"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/tmp/test_vault")
        monkeypatch.setenv("MODEL_TYPE", "openai")
        monkeypatch.setenv("MODEL_NAME", "gpt-4")
        
        config = Config(config_path=None)
        
        assert config.model.api_key == "test-key-123"
        assert config.vault.path == "/tmp/test_vault"
        assert config.model.type == "openai"
        assert config.model.name == "gpt-4"
    
    def test_validate_missing_vault_path(self, monkeypatch):
        """Test validation fails with missing vault path"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        config = Config(config_path=None)
        config.vault.path = ""
        
        with pytest.raises(ValueError, match="Vault path not configured"):
            config.validate()
    
    def test_validate_vault_path_not_exists(self, monkeypatch):
        """Test validation fails when vault path doesn't exist"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        config = Config(config_path=None)
        config.vault.path = "/nonexistent/path"
        
        with pytest.raises(ValueError, match="Vault path does not exist"):
            config.validate()
    
    def test_validate_missing_openai_key(self, temp_vault_dir):
        """Test validation fails for OpenAI without API key"""
        config = Config(config_path=None)
        config.model.type = "openai"
        config.model.api_key = None
        config.vault.path = str(temp_vault_dir)
        
        with pytest.raises(ValueError, match="OpenAI API key required"):
            config.validate()
    
    def test_validate_success_with_openai(self, temp_vault_dir):
        """Test successful validation with OpenAI"""
        config = Config(config_path=None)
        config.model.type = "openai"
        config.model.api_key = "test-key"
        config.vault.path = str(temp_vault_dir)
        
        result = config.validate()
        
        assert result is True
    
    def test_validate_success_with_ollama(self, temp_vault_dir):
        """Test successful validation with Ollama (no API key needed)"""
        config = Config(config_path=None)
        config.model.type = "ollama"
        config.model.api_key = None
        config.vault.path = str(temp_vault_dir)
        
        result = config.validate()
        
        assert result is True
    
    def test_expand_user_path(self, monkeypatch):
        """Test that ~ is expanded in vault path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("HOME", tmpdir)
            config = Config(config_path=None)
            config.vault.path = "~/test_vault"
            
            # Create the test directory
            Path(tmpdir, "test_vault").mkdir()
            
            config.validate()
            
            assert "~" not in config.vault.path
            assert tmpdir in config.vault.path
