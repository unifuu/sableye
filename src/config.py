import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Model configuration"""
    type: str = "openai"  # "openai" or "ollama"
    name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2000

@dataclass
class VaultConfig:
    """Obsidian vault configuration"""
    path: str = ""
    load_days: Optional[int] = 90
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class AgentConfig:
    """Agent configuration"""
    verbose: bool = False
    max_iterations: int = 5
    search_results_limit: int = 5

class Config:
    """Application configuration manager"""
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = config_path
        elif Path("config.local.yaml").exists():
            self.config_path = "config.local.yaml"
        else:
            self.config_path = "config.yaml"
        self.model = ModelConfig()
        self.vault = VaultConfig()
        self.agent = AgentConfig()
        
        self._load_from_file()
        self._load_from_env()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        if not Path(self.config_path).exists():
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # Load model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # Load vault config
        if "vault" in data:
            for key, value in data["vault"].items():
                if hasattr(self.vault, key):
                    setattr(self.vault, key, value)
        
        # Load agent config
        if "agent" in data:
            for key, value in data["agent"].items():
                if hasattr(self.agent, key):
                    setattr(self.agent, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        if api_key := os.getenv("OPENAI_API_KEY"):
            self.model.api_key = api_key
        
        if vault_path := os.getenv("OBSIDIAN_VAULT_PATH"):
            self.vault.path = vault_path
        
        if model_type := os.getenv("MODEL_TYPE"):
            self.model.type = model_type
        
        if model_name := os.getenv("MODEL_NAME"):
            self.model.name = model_name
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.vault.path:
            raise ValueError("Vault path not configured")
        
        # Expand user path and resolve
        expanded_path = Path(self.vault.path).expanduser().resolve()
        self.vault.path = str(expanded_path)
        
        if not expanded_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault.path}")
        
        if self.model.type == "openai" and not self.model.api_key:
            raise ValueError("OpenAI API key required for openai model type")
        
        return True