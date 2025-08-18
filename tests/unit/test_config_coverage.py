#!/usr/bin/env python3
"""
High-coverage tests for configuration system.

This module focuses on achieving high test coverage for all configuration components.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from globule.config.manager import PydanticConfigManager
from globule.config.sources import load_yaml_file, deep_merge
from globule.config.paths import system_config_path, user_config_path
from globule.config.models import EmbeddingConfig, StorageConfig, GlobuleConfig
from globule.config.errors import ConfigError, ConfigValidationError, ConfigFileError


class TestConfigSourcesCoverage:
    """High coverage tests for configuration sources."""

    def test_load_yaml_file_success(self):
        """Test successful YAML file loading."""
        content = "embedding:\n  provider: ollama\n  model: test-model"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(content)
            f.close()
            try:
                result = load_yaml_file(f.name)
                assert result['embedding']['provider'] == 'ollama'
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_nonexistent(self):
        """Test loading nonexistent file returns empty dict."""
        result = load_yaml_file('/nonexistent/path.yaml')
        assert result == {}

    def test_load_yaml_file_empty(self):
        """Test loading empty file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            f.close()
            try:
                result = load_yaml_file(f.name)
                assert result == {}
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_yaml_error(self):
        """Test YAML parsing error handling."""
        content = "invalid: [unclosed"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(content)
            f.close()
            try:
                with pytest.raises(ConfigFileError) as exc_info:
                    load_yaml_file(f.name)
                assert 'Invalid YAML' in str(exc_info.value)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_load_yaml_file_io_error(self):
        """Test IO error handling."""
        # Create a file first, then patch open to raise IOError
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('test: value')
            f.close()
            
            try:
                with patch('builtins.open', side_effect=IOError("Permission denied")):
                    with pytest.raises(ConfigFileError) as exc_info:
                        load_yaml_file(f.name)
                    assert 'Cannot read config file' in str(exc_info.value)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_deep_merge_simple(self):
        """Test simple dictionary merge."""
        base = {'a': 1, 'b': 2}
        override = {'c': 3}
        result = deep_merge(base, override)
        assert result == {'a': 1, 'b': 2, 'c': 3}

    def test_deep_merge_override_values(self):
        """Test value override in merge."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = deep_merge(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_deep_merge_nested_dicts(self):
        """Test nested dictionary merge."""
        base = {'section': {'key1': 'val1', 'key2': 'val2'}}
        override = {'section': {'key2': 'override2', 'key3': 'val3'}}
        result = deep_merge(base, override)
        expected = {'section': {'key1': 'val1', 'key2': 'override2', 'key3': 'val3'}}
        assert result == expected

    def test_deep_merge_list_replacement(self):
        """Test that lists are replaced, not merged."""
        base = {'items': [1, 2, 3]}
        override = {'items': [4, 5]}
        result = deep_merge(base, override)
        assert result == {'items': [4, 5]}

    def test_deep_merge_empty_dicts(self):
        """Test merging with empty dictionaries."""
        assert deep_merge({}, {}) == {}
        assert deep_merge({'a': 1}, {}) == {'a': 1}
        assert deep_merge({}, {'b': 2}) == {'b': 2}


class TestConfigPathsCoverage:
    """High coverage tests for configuration paths."""

    @patch('platform.system', return_value='Windows')
    def test_system_config_path_windows(self, mock_system):
        """Test Windows system config path."""
        with patch.dict(os.environ, {'PROGRAMDATA': 'C:\\ProgramData'}):
            path = system_config_path()
            assert path == Path('C:\\ProgramData\\Globule\\config.yaml')

    @patch('platform.system', return_value='Windows')
    def test_system_config_path_windows_fallback(self, mock_system):
        """Test Windows system config path fallback."""
        with patch.dict(os.environ, {}, clear=True):
            path = system_config_path()
            assert 'Globule' in str(path)
            assert path.name == 'config.yaml'

    @patch('platform.system', return_value='Linux')
    def test_system_config_path_linux(self, mock_system):
        """Test Linux system config path."""
        path = system_config_path()
        assert path == Path('/etc/globule/config.yaml')

    @patch('platform.system', return_value='Darwin')
    def test_system_config_path_macos(self, mock_system):
        """Test macOS system config path."""
        path = system_config_path()
        assert path == Path('/etc/globule/config.yaml')

    @patch('platform.system', return_value='Windows')
    def test_user_config_path_windows(self, mock_system):
        """Test Windows user config path."""
        with patch.dict(os.environ, {'APPDATA': 'C:\\Users\\Test\\AppData\\Roaming'}):
            path = user_config_path()
            assert path == Path('C:\\Users\\Test\\AppData\\Roaming\\Globule\\config.yaml')

    @patch('platform.system', return_value='Windows')
    def test_user_config_path_windows_fallback(self, mock_system):
        """Test Windows user config path fallback."""
        with patch.dict(os.environ, {}, clear=True):
            path = user_config_path()
            assert 'Globule' in str(path)
            assert path.name == 'config.yaml'

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_xdg(self, mock_system):
        """Test Linux user config path with XDG."""
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/custom/config'}):
            path = user_config_path()
            assert path == Path('/custom/config/globule/config.yaml')

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_home(self, mock_system):
        """Test Linux user config path with HOME."""
        with patch.dict(os.environ, {'HOME': '/home/user', 'XDG_CONFIG_HOME': ''}, clear=False):
            path = user_config_path()
            assert path == Path('/home/user/.config/globule/config.yaml')

    @patch('platform.system', return_value='Linux')
    def test_user_config_path_linux_fallback(self, mock_system):
        """Test Linux user config path fallback."""
        with patch.dict(os.environ, {}, clear=True):
            # Mock Path.home() to avoid RuntimeError
            with patch('pathlib.Path.home', return_value=Path('/fallback/home')):
                path = user_config_path()
                assert 'globule' in str(path)
                assert path.name == 'config.yaml'


class TestConfigErrorsCoverage:
    """High coverage tests for configuration errors."""

    def test_config_error_creation(self):
        """Test ConfigError creation."""
        error = ConfigError("Test message")
        assert str(error) == "Test message"
        assert isinstance(error, Exception)

    def test_config_validation_error_creation(self):
        """Test ConfigValidationError creation."""
        error = ConfigValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, ConfigError)

    def test_config_file_error_creation(self):
        """Test ConfigFileError creation."""
        error = ConfigFileError("File error")
        assert str(error) == "File error"
        assert isinstance(error, ConfigError)

    def test_config_file_error_with_source(self):
        """Test ConfigFileError with source attribute."""
        error = ConfigFileError("File error", source="/path/to/file")
        assert str(error) == "File error"
        assert hasattr(error, 'source')
        assert error.source == "/path/to/file"


class TestConfigModelsCoverage:
    """High coverage tests for configuration models."""

    def test_embedding_config_defaults(self):
        """Test EmbeddingConfig default values."""
        config = EmbeddingConfig()
        assert config.provider == 'ollama'
        assert config.model == 'mxbai-embed-large'
        assert config.endpoint is None

    def test_embedding_config_custom_values(self):
        """Test EmbeddingConfig with custom values."""
        config = EmbeddingConfig(
            provider='openai',
            model='text-embedding-3-large',
            endpoint='https://api.openai.com'
        )
        assert config.provider == 'openai'
        assert config.model == 'text-embedding-3-large'
        assert str(config.endpoint) == 'https://api.openai.com/'

    def test_embedding_config_validation_invalid_provider(self):
        """Test EmbeddingConfig provider validation."""
        with pytest.raises(Exception):
            EmbeddingConfig(provider='invalid')

    def test_embedding_config_validation_invalid_endpoint(self):
        """Test EmbeddingConfig endpoint validation."""
        with pytest.raises(Exception):
            EmbeddingConfig(endpoint='http://insecure.com')

    def test_storage_config_defaults(self):
        """Test StorageConfig default values."""
        config = StorageConfig()
        assert config.backend == 'sqlite'
        # Default path is :memory: for testing
        assert config.path == ':memory:'

    def test_storage_config_custom_values(self):
        """Test StorageConfig with custom values."""
        config = StorageConfig(
            backend='postgres',
            path='postgresql://localhost:5432/db'
        )
        assert config.backend == 'postgres'
        assert config.path == 'postgresql://localhost:5432/db'

    def test_storage_config_validation_invalid_backend(self):
        """Test StorageConfig backend validation."""
        with pytest.raises(Exception):
            StorageConfig(backend='invalid')

    def test_globule_config_defaults(self):
        """Test GlobuleConfig default values."""
        config = GlobuleConfig()
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.storage, StorageConfig)

    def test_globule_config_custom_sections(self):
        """Test GlobuleConfig with custom sections."""
        config = GlobuleConfig(
            embedding=EmbeddingConfig(model='custom-model'),
            storage=StorageConfig(backend='postgres')
        )
        assert config.embedding.model == 'custom-model'
        assert config.storage.backend == 'postgres'


class TestPydanticConfigManagerCoverage:
    """High coverage tests for PydanticConfigManager."""

    def test_config_manager_defaults(self):
        """Test PydanticConfigManager with defaults."""
        config = PydanticConfigManager()
        assert config.get('embedding.provider') == 'ollama'
        assert config.get('storage.backend') == 'sqlite'

    def test_config_manager_explicit_overrides(self):
        """Test PydanticConfigManager with explicit overrides."""
        overrides = {
            'embedding': {'model': 'custom-model'},
            'storage': {'path': './custom.db'}
        }
        config = PydanticConfigManager(overrides=overrides)
        assert config.get('embedding.model') == 'custom-model'
        assert config.get('storage.path') == './custom.db'

    def test_config_manager_environment_variables(self):
        """Test PydanticConfigManager with environment variables."""
        os.environ['GLOBULE_EMBEDDING__MODEL'] = 'env-model'
        try:
            config = PydanticConfigManager()
            assert config.get('embedding.model') == 'env-model'
        finally:
            os.environ.pop('GLOBULE_EMBEDDING__MODEL', None)

    def test_config_manager_get_with_default(self):
        """Test get method with default value."""
        config = PydanticConfigManager()
        result = config.get('nonexistent.key', 'default_value')
        assert result == 'default_value'

    def test_config_manager_get_section(self):
        """Test get_section method."""
        config = PydanticConfigManager()
        embedding_section = config.get_section('embedding')
        assert 'provider' in embedding_section
        assert 'model' in embedding_section

    def test_config_manager_get_section_nonexistent(self):
        """Test get_section with nonexistent section."""
        config = PydanticConfigManager()
        result = config.get_section('nonexistent')
        assert result == {}

    def test_config_manager_reload(self):
        """Test reload method."""
        config = PydanticConfigManager()
        # reload() should not raise an exception
        config.reload()

    def test_config_manager_validation_error(self):
        """Test configuration validation error."""
        with pytest.raises(Exception):
            PydanticConfigManager(overrides={
                'embedding': {'provider': 'invalid_provider'}
            })

    def test_config_manager_precedence(self):
        """Test configuration precedence order."""
        os.environ['GLOBULE_STORAGE__BACKEND'] = 'postgres'
        try:
            config = PydanticConfigManager(overrides={
                'embedding': {'provider': 'ollama'}
            })
            # Explicit override should win
            assert config.get('embedding.provider') == 'ollama'
            # Environment should win for storage
            assert config.get('storage.backend') == 'postgres'
        finally:
            os.environ.pop('GLOBULE_STORAGE__BACKEND', None)

    def test_config_manager_get_edge_cases(self):
        """Test get method edge cases."""
        config = PydanticConfigManager()
        
        # Empty key
        assert config.get('', 'default') == 'default'
        
        # None key  
        assert config.get(None, 'default') == 'default'
        
        # Single level key
        provider = config.get('embedding.provider')
        assert provider == 'ollama'

    def test_config_manager_model_config(self):
        """Test model configuration attributes."""
        config = PydanticConfigManager()
        model_config = config.model_config
        assert model_config['env_prefix'] == 'GLOBULE_'
        assert model_config['env_nested_delimiter'] == '__'
        assert model_config['case_sensitive'] == False


class TestFactoriesCoverage:
    """High coverage tests for factory system."""

    def test_embedding_adapter_factory_ollama(self):
        """Test EmbeddingAdapterFactory with Ollama provider."""
        from globule.core.factories import EmbeddingAdapterFactory
        
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'ollama', 'model': 'test-model'}
        })
        
        with patch('globule.services.embedding.ollama_provider.OllamaEmbeddingProvider'), \
             patch('globule.services.embedding.ollama_adapter.OllamaEmbeddingAdapter') as mock_adapter:
            
            mock_adapter.return_value = 'mocked_adapter'
            result = EmbeddingAdapterFactory.create(config)
            assert result == 'mocked_adapter'

    def test_embedding_adapter_factory_unsupported(self):
        """Test EmbeddingAdapterFactory with unsupported provider."""
        from globule.core.factories import EmbeddingAdapterFactory
        from globule.config.errors import ConfigError
        
        # Use valid provider for config creation, then mock to return unsupported
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'ollama'}
        })
        
        # Mock get method to return unsupported provider
        original_get = config.get
        def mock_get(key, default=None):
            if key == 'embedding.provider':
                return 'unsupported'
            return original_get(key, default)
        
        with patch.object(config, 'get', side_effect=mock_get):
            with pytest.raises(ConfigError) as exc_info:
                EmbeddingAdapterFactory.create(config)
            assert 'Unsupported embedding provider' in str(exc_info.value)

    def test_storage_manager_factory_sqlite(self):
        """Test StorageManagerFactory with SQLite backend."""
        from globule.core.factories import StorageManagerFactory
        
        config = PydanticConfigManager(overrides={
            'storage': {'backend': 'sqlite', 'path': './test.db'}
        })
        
        with patch('globule.storage.sqlite_adapter.SqliteStorageAdapter') as mock_adapter:
            mock_adapter.return_value = 'mocked_storage'
            result = StorageManagerFactory.create(config)
            mock_adapter.assert_called_once_with(db_path='./test.db')
            assert result == 'mocked_storage'

    def test_storage_manager_factory_unsupported(self):
        """Test StorageManagerFactory with unsupported backend."""
        from globule.core.factories import StorageManagerFactory
        from globule.config.errors import ConfigError
        
        # Use valid backend for config creation, then mock to return unsupported
        config = PydanticConfigManager(overrides={
            'storage': {'backend': 'sqlite'}
        })
        
        # Mock get method to return unsupported backend
        original_get = config.get
        def mock_get(key, default=None):
            if key == 'storage.backend':
                return 'unsupported'
            return original_get(key, default)
        
        with patch.object(config, 'get', side_effect=mock_get):
            with pytest.raises(ConfigError) as exc_info:
                StorageManagerFactory.create(config)
            assert 'Unsupported storage backend' in str(exc_info.value)

    def test_parser_provider_factory_ollama(self):
        """Test ParserProviderFactory with Ollama provider."""
        from globule.core.factories import ParserProviderFactory
        
        config = PydanticConfigManager()
        
        with patch('globule.services.providers_mock.MockParserProvider') as mock_parser:
            mock_parser.return_value = 'mocked_parser'
            result = ParserProviderFactory.create(config)
            mock_parser.assert_called_once_with()
            assert result == 'mocked_parser'

    def test_orchestrator_factory_create(self):
        """Test OrchestratorFactory create method."""
        from globule.core.factories import OrchestratorFactory
        
        config = PydanticConfigManager()
        
        with patch('globule.core.factories.EmbeddingAdapterFactory.create') as mock_embed, \
             patch('globule.core.factories.StorageManagerFactory.create') as mock_storage, \
             patch('globule.core.factories.ParserProviderFactory.create') as mock_parser, \
             patch('globule.orchestration.engine.GlobuleOrchestrator') as mock_orchestrator:
            
            mock_embed.return_value = 'mock_embedding'
            mock_storage.return_value = 'mock_storage'
            mock_parser.return_value = 'mock_parser'
            mock_orchestrator.return_value = 'mock_orchestrator'
            
            result = OrchestratorFactory.create(config)
            
            mock_orchestrator.assert_called_once_with(
                parser_provider='mock_parser',
                embedding_provider='mock_embedding',
                storage_manager='mock_storage'
            )
            assert result == 'mock_orchestrator'

    def test_create_default_orchestrator(self):
        """Test create_default_orchestrator function."""
        from globule.core.factories import create_default_orchestrator, OrchestratorFactory
        
        with patch.object(OrchestratorFactory, 'create') as mock_create:
            mock_create.return_value = 'mocked_orchestrator'
            
            # Test without overrides
            result = create_default_orchestrator()
            assert result == 'mocked_orchestrator'
            
            # Test with overrides
            overrides = {'embedding': {'model': 'custom'}}
            result = create_default_orchestrator(overrides)
            assert result == 'mocked_orchestrator'
            
            # Test with None overrides
            result = create_default_orchestrator(None)
            assert result == 'mocked_orchestrator'