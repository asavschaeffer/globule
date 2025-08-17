"""
Configuration manager implementing three-tier cascade.

Provides PydanticConfigManager that implements IConfigManager with
defaults + system file cascade and validation.
"""
from typing import Dict, Any, Optional
from pydantic import ValidationError

from globule.core.interfaces import IConfigManager
from .models import GlobuleConfig
from .errors import ConfigError, ConfigValidationError
from .paths import system_config_path, user_config_path
from .sources import load_yaml_file, deep_merge


class PydanticConfigManager(IConfigManager):
    """
    Configuration manager using Pydantic models with cascade resolution.
    
    Implements three-tier cascade: defaults → system file → user file → overrides.
    """
    
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager with cascade resolution.
        
        Args:
            overrides: Optional explicit overrides (highest precedence).
            
        Raises:
            ConfigValidationError: If final configuration is invalid.
            ConfigError: If configuration cannot be loaded.
        """
        try:
            # Start with defaults from Pydantic model
            defaults = GlobuleConfig().model_dump()
            
            # Load and merge system configuration
            system_data = load_yaml_file(system_config_path())
            merged_data = deep_merge(defaults, system_data)
            
            # Load and merge user configuration
            user_data = load_yaml_file(user_config_path())
            merged_data = deep_merge(merged_data, user_data)
            
            # Apply explicit overrides if provided
            if overrides:
                merged_data = deep_merge(merged_data, overrides)
            
            # Validate final configuration
            self._config = GlobuleConfig(**merged_data)
            self._data = self._config.model_dump()
            
        except ValidationError as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to initialize configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'embedding.model').
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
        """
        value = self._data
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'embedding', 'storage').
            
        Returns:
            Dictionary containing all keys in the section.
        """
        return self._data.get(section, {})
    
    def reload(self) -> None:
        """
        Reload configuration from all sources.
        
        Note: For development use only. Production should use immutable configs.
        """
        # Re-initialize with same parameters
        # This is a simplified implementation; could be enhanced to preserve overrides
        self.__init__()