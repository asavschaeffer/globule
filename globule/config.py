"""Configuration management for Globule."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration model for Globule."""
    
    # LLM settings
    llm_provider: str = "local"  # local or cloud
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"
    
    # Embedding settings
    embedding_provider: str = "local"  # local or cloud
    embedding_model: str = "mxbai-embed-large:latest"
    embedding_base_url: str = "http://localhost:11434"
    
    # Database settings
    db_path: str = "globule.db"
    
    # Report settings
    report_template: str = "daily.md"
    
    # Performance settings
    max_cache_size: int = 1000
    embedding_timeout: int = 30
    llm_timeout: int = 60
    
    # API keys (for cloud providers)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_path: str = "config.yaml") -> "Config":
        """Load configuration from a YAML file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            # Create default config file
            config = cls()
            config.save_to_file(config_path)
            return config
        
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Override with environment variables
        config_data = cls._override_with_env(config_data)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: str = "config.yaml"):
        """Save configuration to a YAML file."""
        config_data = self.model_dump()
        
        # Don't save API keys to file for security
        config_data.pop("openai_api_key", None)
        config_data.pop("anthropic_api_key", None)
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    @staticmethod
    def _override_with_env(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        env_mappings = {
            "GLOBULE_LLM_PROVIDER": "llm_provider",
            "GLOBULE_LLM_MODEL": "llm_model",
            "GLOBULE_LLM_BASE_URL": "llm_base_url",
            "GLOBULE_EMBEDDING_PROVIDER": "embedding_provider",
            "GLOBULE_EMBEDDING_MODEL": "embedding_model",
            "GLOBULE_EMBEDDING_BASE_URL": "embedding_base_url",
            "GLOBULE_DB_PATH": "db_path",
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                config_data[config_key] = os.environ[env_var]
        
        return config_data


def load_config() -> Config:
    """Load the global configuration."""
    return Config.from_file()


def create_default_config():
    """Create a default configuration file."""
    config = Config()
    config.save_to_file()
    print("Created default config.yaml file")
    return config