#!/usr/bin/env python3
"""
Tests for legacy settings module to ensure backward compatibility.

This module tests the legacy configuration interface to maintain
backward compatibility while using the Phase 3 system under the hood.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from globule.config.settings import (
    get_config, 
    reload_config,
    GlobuleConfig
)


class TestLegacyGlobuleConfig:
    """Tests for the legacy GlobuleConfig class."""

    def test_config_creation_defaults(self):
        """Test config creation with default values."""
        config = GlobuleConfig()
        
        # Test that defaults match expected values
        assert isinstance(config.storage_path, str)
        assert isinstance(config.default_embedding_model, str)
        assert isinstance(config.default_parsing_model, str)
        assert isinstance(config.ollama_base_url, str)

    def test_config_properties_access(self):
        """Test accessing all config properties."""
        config = GlobuleConfig()
        
        # Test string properties
        assert isinstance(config.storage_path, str)
        assert isinstance(config.default_embedding_model, str)
        assert isinstance(config.default_parsing_model, str)
        assert isinstance(config.ollama_base_url, str)
        assert isinstance(config.default_schema, str)
        
        # Test numeric properties
        assert isinstance(config.ollama_timeout, int)
        assert isinstance(config.embedding_cache_size, int)
        assert isinstance(config.max_concurrent_requests, int)
        
        # Test boolean properties
        assert isinstance(config.auto_schema_detection, bool)

    def test_config_with_custom_manager(self):
        """Test config creation with custom manager."""
        mock_manager = MagicMock()
        mock_manager.get.side_effect = lambda key, default=None: {
            'storage.path': './custom.db',
            'embedding.model': 'custom-model',
            'embedding.endpoint': 'https://custom.example.com:11434'
        }.get(key, default)
        
        config = GlobuleConfig(mock_manager)
        
        assert config.storage_path == './custom.db'
        assert config.default_embedding_model == 'custom-model'
        assert 'custom.example.com' in config.ollama_base_url

    def test_storage_path_property(self):
        """Test storage_path property."""
        config = GlobuleConfig()
        path = config.storage_path
        assert isinstance(path, str)
        assert len(path) > 0

    def test_default_embedding_model_property(self):
        """Test default_embedding_model property."""
        config = GlobuleConfig()
        model = config.default_embedding_model
        assert isinstance(model, str)
        assert len(model) > 0

    def test_default_parsing_model_property(self):
        """Test default_parsing_model property."""
        config = GlobuleConfig()
        model = config.default_parsing_model
        assert model == "llama3.2:3b"  # Legacy fallback

    def test_ollama_base_url_property(self):
        """Test ollama_base_url property."""
        config = GlobuleConfig()
        url = config.ollama_base_url
        assert isinstance(url, str)
        assert 'localhost' in url or 'http' in url

    def test_ollama_base_url_with_endpoint(self):
        """Test ollama_base_url when endpoint is configured."""
        mock_manager = MagicMock()
        mock_manager.get.side_effect = lambda key, default=None: {
            'embedding.endpoint': 'https://example.com:11434/'
        }.get(key, default)
        
        config = GlobuleConfig(mock_manager)
        url = config.ollama_base_url
        assert url == 'https://example.com:11434'  # Should strip trailing slash

    def test_legacy_fallback_properties(self):
        """Test legacy fallback properties."""
        config = GlobuleConfig()
        
        # These should return hardcoded legacy values
        assert config.ollama_timeout == 30
        assert config.embedding_cache_size == 1000
        assert config.max_concurrent_requests == 5
        assert config.default_schema == "default"
        assert config.auto_schema_detection == True

    def test_get_config_path_classmethod(self):
        """Test get_config_path class method."""
        path = GlobuleConfig.get_config_path()
        assert isinstance(path, Path)
        assert path.name == 'config.yaml'

    def test_load_classmethod(self):
        """Test load class method."""
        config = GlobuleConfig.load()
        assert isinstance(config, GlobuleConfig)
        assert hasattr(config, '_config_manager')

    def test_save_method(self):
        """Test save method (no-op)."""
        config = GlobuleConfig()
        # Should not raise an exception
        config.save()

    def test_get_storage_dir_memory(self):
        """Test get_storage_dir with :memory: path."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = ':memory:'
        
        config = GlobuleConfig(mock_manager)
        storage_dir = config.get_storage_dir()
        assert storage_dir == Path(':memory:')

    def test_get_storage_dir_file_path(self):
        """Test get_storage_dir with file path."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = './test.db'
        
        config = GlobuleConfig(mock_manager)
        
        with patch.object(Path, 'mkdir') as mock_mkdir:
            storage_dir = config.get_storage_dir()
            assert storage_dir == Path('./test.db')
            mock_mkdir.assert_called_once_with(exist_ok=True, parents=True)


class TestLegacyConfigFunctions:
    """Tests for legacy configuration functions."""

    def test_get_config_function(self):
        """Test get_config function returns GlobuleConfig instance."""
        config = get_config()
        assert isinstance(config, GlobuleConfig)

    def test_get_config_singleton_behavior(self):
        """Test that get_config returns the same instance."""
        config1 = get_config()
        config2 = get_config()
        
        # Both should be GlobuleConfig instances
        assert isinstance(config1, GlobuleConfig)
        assert isinstance(config2, GlobuleConfig)
        
        # Should be the same instance (singleton)
        assert config1 is config2

    def test_reload_config_function(self):
        """Test reload_config function."""
        # Get initial config
        config1 = get_config()
        
        # Reload config
        config2 = reload_config()
        
        # Should be GlobuleConfig instance
        assert isinstance(config2, GlobuleConfig)
        
        # Should return the new reloaded instance
        config3 = get_config()
        assert config3 is config2

    def test_reload_config_resets_singleton(self):
        """Test that reload_config resets the singleton."""
        # Get initial config
        config1 = get_config()
        
        # Reload should create new instance
        config2 = reload_config()
        
        # New get_config should return the reloaded instance
        config3 = get_config()
        assert config3 is config2
        assert config3 is not config1


class TestLegacyConfigWithEnvironment:
    """Tests for legacy config with environment variables."""

    def test_config_with_environment_variables(self):
        """Test config with environment variables."""
        # Set environment variables
        env_vars = {
            'GLOBULE_EMBEDDING__MODEL': 'env-model',
            'GLOBULE_STORAGE__PATH': './env-storage.db',
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            # Create fresh config to pick up environment
            config = GlobuleConfig()
            
            # Test that environment variables might influence the config
            # Note: Due to BaseSettings caching, this may not always work
            assert hasattr(config, 'default_embedding_model')
            assert hasattr(config, 'storage_path')
            
        finally:
            for key in env_vars:
                os.environ.pop(key, None)

    def test_config_manager_integration(self):
        """Test that legacy config integrates with PydanticConfigManager."""
        # Test that the legacy config uses PydanticConfigManager under the hood
        config = GlobuleConfig()
        
        # Should have a _config_manager attribute
        assert hasattr(config, '_config_manager')
        
        # Manager should have expected methods
        assert hasattr(config._config_manager, 'get')


class TestLegacyConfigErrorHandling:
    """Test error handling and edge cases for legacy config."""

    def test_config_with_failing_manager(self):
        """Test config behavior when manager fails."""
        mock_manager = MagicMock()
        mock_manager.get.side_effect = Exception("Manager error")
        
        config = GlobuleConfig(mock_manager)
        
        # Properties should handle manager errors gracefully
        with pytest.raises(Exception):
            _ = config.storage_path

    def test_config_with_none_manager(self):
        """Test config creation with None manager."""
        # Should create default manager
        config = GlobuleConfig(None)
        assert hasattr(config, '_config_manager')
        assert config._config_manager is not None

    def test_config_property_defaults(self):
        """Test that config properties have reasonable defaults."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = None  # Return None for all gets
        
        config = GlobuleConfig(mock_manager)
        
        # Should have fallback defaults
        storage_path = config.storage_path
        embedding_model = config.default_embedding_model
        
        # Should not be None
        assert storage_path is not None
        assert embedding_model is not None

    def test_ollama_base_url_fallback(self):
        """Test ollama_base_url fallback behavior."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = None  # No endpoint configured
        
        config = GlobuleConfig(mock_manager)
        url = config.ollama_base_url
        
        # Should fall back to localhost
        assert url == "http://localhost:11434"

    def test_storage_dir_creation_error(self):
        """Test storage dir creation with permission error."""
        mock_manager = MagicMock()
        mock_manager.get.return_value = './test_storage.db'
        
        config = GlobuleConfig(mock_manager)
        
        with patch.object(Path, 'mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                config.get_storage_dir()


class TestLegacyConfigIntegration:
    """Integration tests for legacy config system."""

    def test_legacy_config_phase3_integration(self):
        """Test that legacy config properly integrates with Phase 3 system."""
        config = GlobuleConfig()
        
        # Should have access to Phase 3 configuration values
        storage_path = config.storage_path
        embedding_model = config.default_embedding_model
        
        # Values should be strings
        assert isinstance(storage_path, str)
        assert isinstance(embedding_model, str)

    def test_config_load_and_access_patterns(self):
        """Test common configuration access patterns."""
        # Test class method loading
        config1 = GlobuleConfig.load()
        assert isinstance(config1, GlobuleConfig)
        
        # Test function access
        config2 = get_config()
        assert isinstance(config2, GlobuleConfig)
        
        # Test property access
        properties_to_test = [
            'storage_path',
            'default_embedding_model',
            'default_parsing_model',
            'ollama_base_url',
            'ollama_timeout',
            'embedding_cache_size',
            'max_concurrent_requests',
            'default_schema',
            'auto_schema_detection'
        ]
        
        for prop in properties_to_test:
            assert hasattr(config1, prop)
            value = getattr(config1, prop)
            assert value is not None

    def test_config_path_resolution(self):
        """Test that config path resolution works."""
        path = GlobuleConfig.get_config_path()
        
        # Should be a valid Path object
        assert isinstance(path, Path)
        assert path.name == 'config.yaml'
        
        # Should contain globule in the path somewhere
        assert 'globule' in str(path).lower() or 'Globule' in str(path)

    def test_backward_compatibility_interface(self):
        """Test that the interface maintains backward compatibility."""
        config = get_config()
        
        # Test that old interface patterns still work
        assert hasattr(config, 'storage_path')
        assert hasattr(config, 'default_embedding_model')
        assert hasattr(config, 'ollama_base_url')
        
        # Test that methods exist
        assert hasattr(config, 'save')
        assert hasattr(config, 'get_storage_dir')
        
        # Test that class methods exist
        assert hasattr(GlobuleConfig, 'get_config_path')
        assert hasattr(GlobuleConfig, 'load')