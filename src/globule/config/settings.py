"""
Configuration management for Globule MVP.

Follows the simplified approach outlined in the MVP kickoff memo:
- Single config.yaml in user's config directory
- 3-4 simple, essential keys
- No complex cascading or hot-reloading for MVP
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class GlobuleConfig:
    """Main configuration for Globule"""
    
    # Storage settings
    storage_path: str = field(default_factory=lambda: str(Path.home() / ".globule" / "data"))
    
    # AI model settings  
    default_embedding_model: str = "mxbai-embed-large"
    default_parsing_model: str = "llama3.2:3b"
    
    # Ollama connection
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 30
    
    # Performance settings
    embedding_cache_size: int = 1000
    max_concurrent_requests: int = 5
    
    @classmethod
    def get_config_path(cls) -> Path:
        """Get the path to the configuration file"""
        config_dir = Path.home() / ".globule"
        config_dir.mkdir(exist_ok=True)
        return config_dir / "config.yaml"
    
    @classmethod
    def load(cls) -> "GlobuleConfig":
        """Load configuration from file or create default"""
        config_path = cls.get_config_path()
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                return cls(**data)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
                print("Using default configuration.")
        
        # Create default config if none exists
        config = cls()
        config.save()
        return config
    
    def save(self) -> None:
        """Save configuration to file"""
        config_path = self.get_config_path()
        config_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Convert to dict for YAML serialization
        data = {
            'storage_path': self.storage_path,
            'default_embedding_model': self.default_embedding_model,
            'default_parsing_model': self.default_parsing_model,
            'ollama_base_url': self.ollama_base_url,
            'ollama_timeout': self.ollama_timeout,
            'embedding_cache_size': self.embedding_cache_size,
            'max_concurrent_requests': self.max_concurrent_requests,
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_storage_dir(self) -> Path:
        """Get the storage directory as a Path object"""
        path = Path(self.storage_path)
        path.mkdir(exist_ok=True, parents=True)
        return path


# Global config instance
_config: Optional[GlobuleConfig] = None


def get_config() -> GlobuleConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = GlobuleConfig.load()
    return _config


def reload_config() -> GlobuleConfig:
    """Reload configuration from file"""
    global _config
    _config = GlobuleConfig.load()
    return _config