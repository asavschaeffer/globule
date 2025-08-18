#!/usr/bin/env python3
"""
Golden snapshot tests for configuration system.

These tests capture the expected behavior of the configuration system
with known inputs and verify outputs remain consistent.
"""

import os
import tempfile
import pytest
from pathlib import Path
from globule.config.manager import PydanticConfigManager
from globule.config.sources import load_yaml_file, deep_merge
from globule.config.paths import system_config_path, user_config_path


class TestConfigurationGoldenSnapshots:
    """Golden snapshot tests for configuration management."""

    def test_default_configuration_snapshot(self):
        """Test that default configuration produces expected output."""
        config = PydanticConfigManager()
        
        # Golden snapshot - these values should remain stable
        assert config.get('embedding.provider') == 'ollama'
        assert config.get('embedding.model') == 'mxbai-embed-large'
        assert config.get('embedding.endpoint') is None
        assert config.get('storage.backend') == 'sqlite'
        assert config.get('storage.path') == './globule.db'

    def test_environment_variable_snapshot(self):
        """Test environment variable parsing produces expected output."""
        # Set up environment variables
        env_vars = {
            'GLOBULE_EMBEDDING__PROVIDER': 'ollama',
            'GLOBULE_EMBEDDING__MODEL': 'codellama:7b',
            'GLOBULE_STORAGE__BACKEND': 'postgres',
            'GLOBULE_STORAGE__PATH': 'postgresql://localhost:5432/test'
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        try:
            config = PydanticConfigManager()
            
            # Golden snapshot - environment variables override defaults
            assert config.get('embedding.provider') == 'ollama'
            assert config.get('embedding.model') == 'codellama:7b'
            assert config.get('storage.backend') == 'postgres'
            assert config.get('storage.path') == 'postgresql://localhost:5432/test'
            
        finally:
            # Clean up environment
            for key in env_vars:
                os.environ.pop(key, None)

    def test_explicit_override_snapshot(self):
        """Test explicit overrides produce expected output."""
        overrides = {
            'embedding': {
                'provider': 'ollama',
                'model': 'llama3.2:3b',
                'endpoint': 'https://localhost:11434'
            },
            'storage': {
                'backend': 'sqlite',
                'path': '/tmp/test.db'
            }
        }
        
        config = PydanticConfigManager(overrides=overrides)
        
        # Golden snapshot - explicit overrides take precedence
        assert config.get('embedding.provider') == 'ollama'
        assert config.get('embedding.model') == 'llama3.2:3b'
        assert config.get('embedding.endpoint') == 'https://localhost:11434'
        assert config.get('storage.backend') == 'sqlite'
        assert config.get('storage.path') == '/tmp/test.db'

    def test_yaml_file_snapshot(self):
        """Test YAML file loading produces expected output."""
        yaml_content = """
embedding:
  provider: ollama
  model: mxbai-embed-large
  endpoint: https://example.com:11434

storage:
  backend: postgres
  path: postgresql://user:pass@localhost:5432/globule
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                result = load_yaml_file(f.name)
                
                # Golden snapshot - YAML parsing output
                expected = {
                    'embedding': {
                        'provider': 'ollama',
                        'model': 'mxbai-embed-large',
                        'endpoint': 'https://example.com:11434'
                    },
                    'storage': {
                        'backend': 'postgres',
                        'path': 'postgresql://user:pass@localhost:5432/globule'
                    }
                }
                
                assert result == expected
                
            finally:
                os.unlink(f.name)

    def test_deep_merge_snapshot(self):
        """Test deep merge produces expected output."""
        dict1 = {
            'embedding': {
                'provider': 'ollama',
                'model': 'default-model'
            },
            'storage': {
                'backend': 'sqlite'
            }
        }
        
        dict2 = {
            'embedding': {
                'model': 'overridden-model',
                'endpoint': 'https://localhost:11434'
            },
            'storage': {
                'path': '/custom/path.db'
            }
        }
        
        result = deep_merge(dict1, dict2)
        
        # Golden snapshot - deep merge preserves and overrides correctly
        expected = {
            'embedding': {
                'provider': 'ollama',
                'model': 'overridden-model',
                'endpoint': 'https://localhost:11434'
            },
            'storage': {
                'backend': 'sqlite',
                'path': '/custom/path.db'
            }
        }
        
        assert result == expected

    def test_get_section_snapshot(self):
        """Test get_section produces expected output."""
        overrides = {
            'embedding': {
                'provider': 'ollama',
                'model': 'mxbai-embed-large',
                'endpoint': 'https://localhost:11434'
            },
            'storage': {
                'backend': 'postgres',
                'path': 'postgresql://localhost/db'
            }
        }
        
        config = PydanticConfigManager(overrides=overrides)
        
        # Golden snapshot - section extraction
        embedding_section = config.get_section('embedding')
        storage_section = config.get_section('storage')
        
        assert embedding_section == {
            'provider': 'ollama',
            'model': 'mxbai-embed-large',
            'endpoint': 'https://localhost:11434'
        }
        
        assert storage_section == {
            'backend': 'postgres',
            'path': 'postgresql://localhost/db'
        }

    def test_precedence_cascade_snapshot(self):
        """Test full precedence cascade produces expected output."""
        # Set up environment variable
        os.environ['GLOBULE_EMBEDDING__MODEL'] = 'env-model'
        
        # Override with explicit value
        overrides = {
            'embedding': {
                'provider': 'explicit-provider'
            }
        }
        
        try:
            config = PydanticConfigManager(overrides=overrides)
            
            # Golden snapshot - precedence order verification
            # Explicit override wins for provider
            assert config.get('embedding.provider') == 'explicit-provider'
            # Environment variable wins for model
            assert config.get('embedding.model') == 'env-model'
            # Default wins for endpoint (no override or env)
            assert config.get('embedding.endpoint') is None
            
        finally:
            os.environ.pop('GLOBULE_EMBEDDING__MODEL', None)

    def test_validation_error_snapshot(self):
        """Test validation errors produce expected output."""
        # Invalid provider
        with pytest.raises(Exception) as exc_info:
            PydanticConfigManager(overrides={
                'embedding': {'provider': 'invalid-provider'}
            })
        
        # Golden snapshot - validation error should mention allowed values
        error_str = str(exc_info.value)
        assert 'ollama' in error_str or 'openai' in error_str or 'huggingface' in error_str

    def test_path_resolution_snapshot(self):
        """Test path resolution produces expected output."""
        # These tests verify path construction is stable across platforms
        # Note: Actual paths will vary by platform, so we test structure
        
        system_path = system_config_path()
        user_path = user_config_path()
        
        # Golden snapshot - paths should be Path objects and end with config.yaml
        assert isinstance(system_path, Path)
        assert isinstance(user_path, Path)
        assert system_path.name == 'config.yaml'
        assert user_path.name == 'config.yaml'
        
        # Platform-specific validations
        import platform
        if platform.system() == 'Windows':
            assert 'ProgramData' in str(system_path) or 'AppData' in str(user_path)
        else:
            assert '/etc' in str(system_path) or '/.config' in str(user_path)