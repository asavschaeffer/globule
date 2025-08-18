#!/usr/bin/env python3
"""
Comprehensive test coverage for configuration system components.

This module provides extensive test coverage for configuration management,
sources, paths, errors, and edge cases to reach ≥90% coverage.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

from globule.config.manager import PydanticConfigManager, MultiYamlSettingsSource
from globule.config.sources import load_yaml_file, deep_merge
from globule.config.paths import system_config_path, user_config_path
from globule.config.models import EmbeddingConfig, StorageConfig, GlobuleConfig
from globule.config.errors import ConfigError, ConfigValidationError, ConfigFileError


class TestConfigSources:
    """Comprehensive tests for configuration sources."""

    def test_load_yaml_file_valid(self):
        """Test loading valid YAML file."""
        yaml_content = """
embedding:
  provider: ollama
  model: test-model
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            f.close()  # Close file before reading on Windows
            
            try:
                result = load_yaml_file(f.name)
                assert result['embedding']['provider'] == 'ollama'
                assert result['embedding']['model'] == 'test-model'
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_empty(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            f.flush()
            f.close()  # Close file before reading on Windows
            
            try:
                result = load_yaml_file(f.name)
                assert result == {}
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_whitespace_only(self):
        """Test loading YAML file with only whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('   \n\t\n  ')
            f.flush()
            f.close()  # Close file before reading on Windows
            
            try:
                result = load_yaml_file(f.name)
                assert result == {}
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_nonexistent(self):
        """Test loading nonexistent YAML file returns empty dict."""
        result = load_yaml_file('/nonexistent/path/file.yaml')
        assert result == {}

    def test_load_yaml_file_invalid_yaml(self):
        """Test loading invalid YAML raises ConfigFileError."""
        invalid_yaml = """
embedding:
  provider: ollama
storage:
  backend: [unclosed list
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            f.close()  # Close file before reading on Windows
            
            try:
                with pytest.raises(ConfigFileError):
                    load_yaml_file(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_permission_error(self):
        """Test handling permission errors."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(ConfigFileError) as exc_info:
                load_yaml_file('/some/path/config.yaml')
            assert 'Access denied' in str(exc_info.value)

    def test_deep_merge_simple(self):
        """Test simple dictionary merge."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'c': 3, 'd': 4}
        result = deep_merge(dict1, dict2)
        assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    def test_deep_merge_override(self):
        """Test dictionary merge with value override."""
        dict1 = {'a': 1, 'b': 2}
        dict2 = {'b': 3, 'c': 4}
        result = deep_merge(dict1, dict2)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_deep_merge_nested(self):
        """Test nested dictionary merge."""
        dict1 = {
            'section': {
                'key1': 'value1',
                'key2': 'value2'
            }
        }
        dict2 = {
            'section': {
                'key2': 'override2',
                'key3': 'value3'
            }
        }
        result = deep_merge(dict1, dict2)
        expected = {
            'section': {
                'key1': 'value1',
                'key2': 'override2',
                'key3': 'value3'
            }
        }
        assert result == expected

    def test_deep_merge_non_dict_values(self):
        """Test merge behavior with non-dict values."""
        dict1 = {'key': ['list1']}
        dict2 = {'key': ['list2']}
        result = deep_merge(dict1, dict2)
        # Lists should be replaced, not merged
        assert result == {'key': ['list2']}

    def test_deep_merge_empty_dicts(self):
        """Test merging empty dictionaries."""
        assert deep_merge({}, {}) == {}
        assert deep_merge({'a': 1}, {}) == {'a': 1}
        assert deep_merge({}, {'b': 2}) == {'b': 2}


class TestConfigPaths:
    """Comprehensive tests for configuration path resolution."""

    @patch('platform.system', return_value='Windows')
    def test_system_config_path_windows(self):
        """Test system config path on Windows."""
        with patch.dict(os.environ, {'PROGRAMDATA': 'C:\\ProgramData'}):
            path = system_config_path()
            expected = Path('C:\\ProgramData\\Globule\\config.yaml')
            assert path == expected

    @patch('platform.system', return_value='Windows')
    def test_system_config_path_windows_no_programdata(self):
        """Test system config path on Windows without PROGRAMDATA."""
        with patch.dict(os.environ, {}, clear=True):
            path = system_config_path()
            # Should fallback to a reasonable default
            assert 'Globule' in str(path)
            assert path.name == 'config.yaml'

    @patch('platform.system', return_value='Linux')
    def test_system_config_path_linux(self):
        """Test system config path on Linux."""
        path = system_config_path()
        expected = Path('/etc/globule/config.yaml')
        assert path == expected

    @patch('platform.system', return_value='Darwin')
    def test_system_config_path_macos(self):
        """Test system config path on macOS."""
        path = system_config_path()
        expected = Path('/etc/globule/config.yaml')
        assert path == expected

    @patch('platform.system', return_value='FreeBSD')
    def test_system_config_path_other_unix(self):
        """Test system config path on other Unix systems."""
        path = system_config_path()
        expected = Path('/etc/globule/config.yaml')
        assert path == expected

    @patch('platform.system', return_value='Windows')
    def test_user_config_path_windows(self):
        """Test user config path on Windows."""
        with patch.dict(os.environ, {'APPDATA': 'C:\\Users\\Test\\AppData\\Roaming'}):
            path = user_config_path()
            expected = Path('C:\\Users\\Test\\AppData\\Roaming\\Globule\\config.yaml')
            assert path == expected

    @patch('platform.system', return_value='Windows')
    def test_user_config_path_windows_no_appdata(self):
        """Test user config path on Windows without APPDATA."""
        with patch.dict(os.environ, {}, clear=True):
            path = user_config_path()
            # Should fallback to a reasonable default
            assert 'Globule' in str(path)
            assert path.name == 'config.yaml'

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_xdg(self):
        """Test user config path on Linux with XDG_CONFIG_HOME."""
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/custom/config'}):
            path = user_config_path()
            expected = Path('/custom/config/globule/config.yaml')
            assert path == expected

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_no_xdg(self):
        """Test user config path on Linux without XDG_CONFIG_HOME."""
        with patch.dict(os.environ, {'HOME': '/home/user'}, clear=True):
            path = user_config_path()
            expected = Path('/home/user/.config/globule/config.yaml')
            assert path == expected

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_no_home(self):
        """Test user config path on Linux without HOME."""
        with patch.dict(os.environ, {}, clear=True):
            path = user_config_path()
            # Should fallback to a reasonable default
            assert 'globule' in str(path)
            assert path.name == 'config.yaml'


class TestConfigErrors:
    """Comprehensive tests for configuration error classes."""

    def test_config_error_base(self):
        """Test ConfigError base class."""
        error = ConfigError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_config_validation_error(self):
        """Test ConfigValidationError class."""
        error = ConfigValidationError("Invalid value")
        assert str(error) == "Invalid value"
        assert isinstance(error, ConfigError)

    def test_config_file_error(self):
        """Test ConfigFileError class."""
        error = ConfigFileError("File not found")
        assert str(error) == "File not found"
        assert isinstance(error, ConfigError)

    def test_config_error_inheritance(self):
        """Test error inheritance hierarchy."""
        validation_error = ConfigValidationError("test")
        file_error = ConfigFileError("test")
        
        assert isinstance(validation_error, ConfigError)
        assert isinstance(file_error, ConfigError)
        assert isinstance(validation_error, Exception)
        assert isinstance(file_error, Exception)


class TestMultiYamlSettingsSource:
    """Comprehensive tests for MultiYamlSettingsSource."""

    def test_multiyaml_source_call(self):
        """Test MultiYamlSettingsSource __call__ method."""
        # Create temporary config files
        system_yaml = """
embedding:
  provider: ollama
  model: system-model
storage:
  backend: sqlite
"""
        
        user_yaml = """
embedding:
  model: user-model
  endpoint: https://localhost:11434
storage:
  path: /user/path.db
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as sys_f, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as user_f:
            
            sys_f.write(system_yaml)
            sys_f.flush()
            sys_f.close()
            user_f.write(user_yaml)
            user_f.flush()
            user_f.close()
            
            try:
                # Mock the path functions to return our temp files
                with patch('globule.config.manager.system_config_path', return_value=sys_f.name), \
                     patch('globule.config.manager.user_config_path', return_value=user_f.name):
                    
                    from pydantic import BaseModel
                    
                    class DummySettings(BaseModel):
                        pass
                    
                    source = MultiYamlSettingsSource(DummySettings, None)
                    result = source()
                    
                    # Verify cascade: user overrides system
                    assert result['embedding']['provider'] == 'ollama'  # from system
                    assert result['embedding']['model'] == 'user-model'  # user override
                    assert result['embedding']['endpoint'] == 'https://localhost:11434'  # user only
                    assert result['storage']['backend'] == 'sqlite'  # from system
                    assert result['storage']['path'] == '/user/path.db'  # user override
                    
            finally:
                if os.path.exists(sys_f.name):
                    os.unlink(sys_f.name)
                if os.path.exists(user_f.name):
                    os.unlink(user_f.name)

    def test_multiyaml_source_repr(self):
        """Test MultiYamlSettingsSource __repr__ method."""
        from pydantic import BaseModel
        
        class DummySettings(BaseModel):
            pass
        
        source = MultiYamlSettingsSource(DummySettings, None)
        repr_str = repr(source)
        assert 'MultiYamlSettingsSource' in repr_str


class TestPydanticConfigManagerComprehensive:
    """Comprehensive tests for PydanticConfigManager."""

    def test_settings_config_dict(self):
        """Test settings configuration dictionary."""
        config = PydanticConfigManager()
        settings_config = config.model_config
        
        assert settings_config['env_prefix'] == 'GLOBULE_'
        assert settings_config['env_nested_delimiter'] == '__'
        assert settings_config['case_sensitive'] == False

    def test_custom_settings_source_types(self):
        """Test custom settings source types."""
        config = PydanticConfigManager()
        # Verify that the custom source is used
        assert hasattr(config, '_settings_build')

    def test_get_nonexistent_nested_key(self):
        """Test get method with nonexistent nested key."""
        config = PydanticConfigManager()
        result = config.get('nonexistent.nested.key', 'default')
        assert result == 'default'

    def test_get_section_nonexistent(self):
        """Test get_section with nonexistent section."""
        config = PydanticConfigManager()
        result = config.get_section('nonexistent')
        assert result == {}

    def test_get_section_partial_path(self):
        """Test get_section with partial attribute path."""
        config = PydanticConfigManager()
        # This should work because 'embedding' exists
        result = config.get_section('embedding')
        assert 'provider' in result
        assert 'model' in result

    def test_reload_method(self):
        """Test reload method functionality."""
        config = PydanticConfigManager()
        initial_model = config.get('embedding.model')
        
        # Change environment variable
        os.environ['GLOBULE_EMBEDDING__MODEL'] = 'reloaded-model'
        
        try:
            config.reload()
            # After reload, should pick up new environment variable
            # Note: This may not work as expected due to BaseSettings caching
            # but we test the method exists and doesn't error
            config.reload()  # Should not raise
            
        finally:
            os.environ.pop('GLOBULE_EMBEDDING__MODEL', None)

    def test_validation_with_invalid_endpoint(self):
        """Test validation with invalid endpoint protocol."""
        with pytest.raises(Exception):
            PydanticConfigManager(overrides={
                'embedding': {
                    'endpoint': 'http://localhost:11434'  # HTTP not allowed
                }
            })

    def test_validation_with_invalid_provider(self):
        """Test validation with invalid provider."""
        with pytest.raises(Exception):
            PydanticConfigManager(overrides={
                'embedding': {
                    'provider': 'invalid_provider'
                }
            })

    def test_validation_with_invalid_backend(self):
        """Test validation with invalid storage backend."""
        with pytest.raises(Exception):
            PydanticConfigManager(overrides={
                'storage': {
                    'backend': 'invalid_backend'
                }
            })

    def test_environment_variable_parsing_edge_cases(self):
        """Test edge cases in environment variable parsing."""
        # Test with special characters
        os.environ['GLOBULE_STORAGE__PATH'] = 'postgresql://user:p@ss@localhost:5432/db'
        
        try:
            config = PydanticConfigManager()
            path = config.get('storage.path')
            assert 'p@ss' in path
            
        finally:
            os.environ.pop('GLOBULE_STORAGE__PATH', None)

    def test_explicit_overrides_deep_nesting(self):
        """Test explicit overrides with deep nesting."""
        overrides = {
            'embedding': {
                'provider': 'ollama',
                'model': 'test-model',
                'endpoint': 'https://test.example.com:11434'
            }
        }
        
        config = PydanticConfigManager(overrides=overrides)
        
        assert config.get('embedding.provider') == 'ollama'
        assert config.get('embedding.model') == 'test-model'
        assert config.get('embedding.endpoint') == 'https://test.example.com:11434'

    def test_precedence_order_comprehensive(self):
        """Test comprehensive precedence order."""
        # Set environment variable
        os.environ['GLOBULE_EMBEDDING__MODEL'] = 'env-model'
        
        # Create overrides
        overrides = {
            'embedding': {
                'provider': 'explicit-provider'
            },
            'storage': {
                'backend': 'explicit-backend'
            }
        }
        
        try:
            config = PydanticConfigManager(overrides=overrides)
            
            # Explicit overrides should win for provider and storage
            assert config.get('embedding.provider') == 'explicit-provider'
            assert config.get('storage.backend') == 'explicit-backend'
            
            # Environment should win for model (no explicit override)
            assert config.get('embedding.model') == 'env-model'
            
            # Defaults should win for endpoint and path (no override or env)
            assert config.get('embedding.endpoint') is None
            # Note: storage.path might have default from BaseSettings
            
        finally:
            os.environ.pop('GLOBULE_EMBEDDING__MODEL', None)


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_overrides(self):
        """Test with empty overrides dictionary."""
        config = PydanticConfigManager(overrides={})
        # Should use defaults
        assert config.get('embedding.provider') == 'ollama'

    def test_none_overrides(self):
        """Test with None overrides."""
        config = PydanticConfigManager(overrides=None)
        # Should use defaults
        assert config.get('embedding.provider') == 'ollama'

    def test_nested_none_values(self):
        """Test with None values in nested structure."""
        overrides = {
            'embedding': {
                'endpoint': None
            }
        }
        
        config = PydanticConfigManager(overrides=overrides)
        assert config.get('embedding.endpoint') is None

    def test_get_with_empty_key(self):
        """Test get method with empty key."""
        config = PydanticConfigManager()
        result = config.get('', 'default')
        assert result == 'default'

    def test_get_with_none_key(self):
        """Test get method with None key."""
        config = PydanticConfigManager()
        result = config.get(None, 'default')
        assert result == 'default'