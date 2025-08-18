#!/usr/bin/env python3
"""
Comprehensive test coverage for factory system.

This module provides extensive test coverage for all factory classes
and configuration-driven dependency injection to reach ≥90% coverage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from globule.config.manager import PydanticConfigManager
from globule.core.factories import (
    EmbeddingAdapterFactory,
    StorageManagerFactory,
    ParserProviderFactory,
    OrchestratorFactory,
    create_default_orchestrator
)
from globule.config.errors import ConfigError


class TestEmbeddingAdapterFactoryComprehensive:
    """Comprehensive tests for EmbeddingAdapterFactory."""

    def test_create_ollama_adapter_all_params(self):
        """Test creating Ollama adapter with all parameters."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'model': 'custom-model',
                'endpoint': 'https://custom.example.com:11434'
            }
        })
        
        with patch('globule.core.factories.OllamaEmbeddingProvider') as mock_provider, \
             patch('globule.core.factories.OllamaEmbeddingAdapter') as mock_adapter:
            
            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            result = EmbeddingAdapterFactory.create(config)
            
            # Verify provider created with correct parameters
            mock_provider.assert_called_once_with(
                base_url='https://custom.example.com:11434',
                model='custom-model',
                timeout=30
            )
            
            # Verify adapter created with provider
            mock_adapter.assert_called_once_with(mock_provider_instance)
            
            assert result == mock_adapter_instance

    def test_create_ollama_adapter_minimal_params(self):
        """Test creating Ollama adapter with minimal parameters."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'model': 'minimal-model'
                # No endpoint specified
            }
        })
        
        with patch('globule.core.factories.OllamaEmbeddingProvider') as mock_provider, \
             patch('globule.core.factories.OllamaEmbeddingAdapter') as mock_adapter:
            
            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            result = EmbeddingAdapterFactory.create(config)
            
            # Verify provider created with None endpoint
            mock_provider.assert_called_once_with(
                base_url=None,
                model='minimal-model',
                timeout=30
            )
            
            assert result == mock_adapter_instance

    def test_create_openai_adapter_not_implemented(self):
        """Test that OpenAI adapter raises NotImplementedError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'openai',
                'model': 'text-embedding-3-large'
            }
        })
        
        with pytest.raises(NotImplementedError) as exc_info:
            EmbeddingAdapterFactory.create(config)
        
        assert 'OpenAI embedding adapter not yet implemented' in str(exc_info.value)

    def test_create_huggingface_adapter_not_implemented(self):
        """Test that Hugging Face adapter raises NotImplementedError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'huggingface',
                'model': 'sentence-transformers/all-MiniLM-L6-v2'
            }
        })
        
        with pytest.raises(NotImplementedError) as exc_info:
            EmbeddingAdapterFactory.create(config)
        
        assert 'HuggingFace embedding adapter not yet implemented' in str(exc_info.value)

    def test_create_unsupported_provider(self):
        """Test that unsupported provider raises ConfigError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'unsupported',
                'model': 'test-model'
            }
        })
        
        with pytest.raises(ConfigError) as exc_info:
            EmbeddingAdapterFactory.create(config)
        
        assert 'Unsupported embedding provider: unsupported' in str(exc_info.value)

    def test_import_error_handling(self):
        """Test handling of import errors during adapter creation."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'model': 'test-model'
            }
        })
        
        with patch('globule.core.factories.OllamaEmbeddingProvider', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError):
                EmbeddingAdapterFactory.create(config)


class TestStorageManagerFactoryComprehensive:
    """Comprehensive tests for StorageManagerFactory."""

    def test_create_sqlite_manager_with_path(self):
        """Test creating SQLite manager with custom path."""
        config = PydanticConfigManager(overrides={
            'storage': {
                'backend': 'sqlite',
                'path': '/custom/path/database.db'
            }
        })
        
        with patch('globule.core.factories.SqliteStorageAdapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            result = StorageManagerFactory.create(config)
            
            mock_adapter.assert_called_once_with('/custom/path/database.db')
            assert result == mock_adapter_instance

    def test_create_sqlite_manager_default_path(self):
        """Test creating SQLite manager with default path."""
        config = PydanticConfigManager()  # Uses defaults
        
        with patch('globule.core.factories.SqliteStorageAdapter') as mock_adapter:
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            result = StorageManagerFactory.create(config)
            
            # Should use default path
            mock_adapter.assert_called_once_with('./globule.db')
            assert result == mock_adapter_instance

    def test_create_postgres_manager_not_implemented(self):
        """Test that PostgreSQL manager raises NotImplementedError."""
        config = PydanticConfigManager(overrides={
            'storage': {
                'backend': 'postgres',
                'path': 'postgresql://localhost:5432/globule'
            }
        })
        
        with pytest.raises(NotImplementedError) as exc_info:
            StorageManagerFactory.create(config)
        
        assert 'PostgreSQL storage adapter not yet implemented' in str(exc_info.value)

    def test_create_unsupported_backend(self):
        """Test that unsupported backend raises ConfigError."""
        config = PydanticConfigManager(overrides={
            'storage': {
                'backend': 'unsupported',
                'path': '/some/path'
            }
        })
        
        with pytest.raises(ConfigError) as exc_info:
            StorageManagerFactory.create(config)
        
        assert 'Unsupported storage backend: unsupported' in str(exc_info.value)

    def test_import_error_handling(self):
        """Test handling of import errors during storage creation."""
        config = PydanticConfigManager(overrides={
            'storage': {
                'backend': 'sqlite',
                'path': './test.db'
            }
        })
        
        with patch('globule.core.factories.SqliteStorageAdapter', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError):
                StorageManagerFactory.create(config)


class TestParserProviderFactoryComprehensive:
    """Comprehensive tests for ParserProviderFactory."""

    def test_create_ollama_parser_with_endpoint(self):
        """Test creating Ollama parser with custom endpoint."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'ollama',
                'endpoint': 'https://custom.example.com:11434'
            }
        })
        
        with patch('globule.core.factories.OllamaParser') as mock_parser:
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance
            
            result = ParserProviderFactory.create(config)
            
            mock_parser.assert_called_once_with(
                base_url='https://custom.example.com:11434'
            )
            assert result == mock_parser_instance

    def test_create_ollama_parser_default_endpoint(self):
        """Test creating Ollama parser with default endpoint."""
        config = PydanticConfigManager()  # Uses defaults
        
        with patch('globule.core.factories.OllamaParser') as mock_parser:
            mock_parser_instance = Mock()
            mock_parser.return_value = mock_parser_instance
            
            result = ParserProviderFactory.create(config)
            
            mock_parser.assert_called_once_with(base_url=None)
            assert result == mock_parser_instance

    def test_create_openai_parser_not_implemented(self):
        """Test that OpenAI parser raises NotImplementedError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'openai'
            }
        })
        
        with pytest.raises(NotImplementedError) as exc_info:
            ParserProviderFactory.create(config)
        
        assert 'OpenAI parser not yet implemented' in str(exc_info.value)

    def test_create_huggingface_parser_not_implemented(self):
        """Test that Hugging Face parser raises NotImplementedError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'huggingface'
            }
        })
        
        with pytest.raises(NotImplementedError) as exc_info:
            ParserProviderFactory.create(config)
        
        assert 'HuggingFace parser not yet implemented' in str(exc_info.value)

    def test_create_unsupported_provider(self):
        """Test that unsupported provider raises ConfigError."""
        config = PydanticConfigManager(overrides={
            'embedding': {
                'provider': 'unsupported'
            }
        })
        
        with pytest.raises(ConfigError) as exc_info:
            ParserProviderFactory.create(config)
        
        assert 'Unsupported parsing provider: unsupported' in str(exc_info.value)

    def test_import_error_handling(self):
        """Test handling of import errors during parser creation."""
        config = PydanticConfigManager()
        
        with patch('globule.core.factories.OllamaParser', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError):
                ParserProviderFactory.create(config)


class TestOrchestratorFactoryComprehensive:
    """Comprehensive tests for OrchestratorFactory."""

    def test_create_orchestrator_success(self):
        """Test successful orchestrator creation."""
        config = PydanticConfigManager()
        
        mock_embedding_adapter = Mock()
        mock_storage_manager = Mock()
        mock_parser_provider = Mock()
        mock_orchestrator = Mock()
        
        with patch.object(EmbeddingAdapterFactory, 'create', return_value=mock_embedding_adapter), \
             patch.object(StorageManagerFactory, 'create', return_value=mock_storage_manager), \
             patch.object(ParserProviderFactory, 'create', return_value=mock_parser_provider), \
             patch('globule.core.factories.GlobuleOrchestrator', return_value=mock_orchestrator):
            
            result = OrchestratorFactory.create(config)
            
            assert result == mock_orchestrator

    def test_create_orchestrator_embedding_error(self):
        """Test orchestrator creation with embedding adapter error."""
        config = PydanticConfigManager()
        
        with patch.object(EmbeddingAdapterFactory, 'create', side_effect=Exception("Embedding error")):
            with pytest.raises(Exception) as exc_info:
                OrchestratorFactory.create(config)
            
            assert "Embedding error" in str(exc_info.value)

    def test_create_orchestrator_storage_error(self):
        """Test orchestrator creation with storage manager error."""
        config = PydanticConfigManager()
        
        mock_embedding_adapter = Mock()
        
        with patch.object(EmbeddingAdapterFactory, 'create', return_value=mock_embedding_adapter), \
             patch.object(StorageManagerFactory, 'create', side_effect=Exception("Storage error")):
            
            with pytest.raises(Exception) as exc_info:
                OrchestratorFactory.create(config)
            
            assert "Storage error" in str(exc_info.value)

    def test_create_orchestrator_parser_error(self):
        """Test orchestrator creation with parser provider error."""
        config = PydanticConfigManager()
        
        mock_embedding_adapter = Mock()
        mock_storage_manager = Mock()
        
        with patch.object(EmbeddingAdapterFactory, 'create', return_value=mock_embedding_adapter), \
             patch.object(StorageManagerFactory, 'create', return_value=mock_storage_manager), \
             patch.object(ParserProviderFactory, 'create', side_effect=Exception("Parser error")):
            
            with pytest.raises(Exception) as exc_info:
                OrchestratorFactory.create(config)
            
            assert "Parser error" in str(exc_info.value)

    def test_create_orchestrator_instantiation_error(self):
        """Test orchestrator creation with instantiation error."""
        config = PydanticConfigManager()
        
        mock_embedding_adapter = Mock()
        mock_storage_manager = Mock()
        mock_parser_provider = Mock()
        
        with patch.object(EmbeddingAdapterFactory, 'create', return_value=mock_embedding_adapter), \
             patch.object(StorageManagerFactory, 'create', return_value=mock_storage_manager), \
             patch.object(ParserProviderFactory, 'create', return_value=mock_parser_provider), \
             patch('globule.core.factories.GlobuleOrchestrator', side_effect=Exception("Orchestrator error")):
            
            with pytest.raises(Exception) as exc_info:
                OrchestratorFactory.create(config)
            
            assert "Orchestrator error" in str(exc_info.value)


class TestCreateDefaultOrchestratorComprehensive:
    """Comprehensive tests for create_default_orchestrator function."""

    def test_create_default_orchestrator_no_overrides(self):
        """Test creating default orchestrator without overrides."""
        mock_orchestrator = Mock()
        
        with patch.object(OrchestratorFactory, 'create', return_value=mock_orchestrator) as mock_create:
            result = create_default_orchestrator()
            
            # Should be called with a PydanticConfigManager instance
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], PydanticConfigManager)
            
            assert result == mock_orchestrator

    def test_create_default_orchestrator_with_overrides(self):
        """Test creating default orchestrator with configuration overrides."""
        overrides = {
            'embedding': {
                'provider': 'ollama',
                'model': 'custom-model'
            }
        }
        
        mock_orchestrator = Mock()
        
        with patch.object(OrchestratorFactory, 'create', return_value=mock_orchestrator) as mock_create:
            result = create_default_orchestrator(overrides)
            
            # Should be called with a PydanticConfigManager instance with overrides
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert len(call_args) == 1
            config = call_args[0]
            assert isinstance(config, PydanticConfigManager)
            assert config.get('embedding.model') == 'custom-model'
            
            assert result == mock_orchestrator

    def test_create_default_orchestrator_none_overrides(self):
        """Test creating default orchestrator with None overrides."""
        mock_orchestrator = Mock()
        
        with patch.object(OrchestratorFactory, 'create', return_value=mock_orchestrator) as mock_create:
            result = create_default_orchestrator(None)
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], PydanticConfigManager)
            
            assert result == mock_orchestrator

    def test_create_default_orchestrator_empty_overrides(self):
        """Test creating default orchestrator with empty overrides."""
        mock_orchestrator = Mock()
        
        with patch.object(OrchestratorFactory, 'create', return_value=mock_orchestrator) as mock_create:
            result = create_default_orchestrator({})
            
            mock_create.assert_called_once()
            call_args = mock_create.call_args[0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], PydanticConfigManager)
            
            assert result == mock_orchestrator

    def test_create_default_orchestrator_propagates_errors(self):
        """Test that create_default_orchestrator propagates factory errors."""
        overrides = {
            'embedding': {
                'provider': 'unsupported'
            }
        }
        
        with patch.object(OrchestratorFactory, 'create', side_effect=ConfigError("Unsupported provider")):
            with pytest.raises(ConfigError) as exc_info:
                create_default_orchestrator(overrides)
            
            assert "Unsupported provider" in str(exc_info.value)


class TestFactoryIntegrationEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_factory_with_complex_configuration(self):
        """Test factory behavior with complex configuration."""
        complex_config = {
            'embedding': {
                'provider': 'ollama',
                'model': 'llama3.2:1b',
                'endpoint': 'https://complex.example.com:11434'
            },
            'storage': {
                'backend': 'sqlite',
                'path': './complex-test.db'
            }
        }
        
        config = PydanticConfigManager(overrides=complex_config)
        
        # Test each factory individually
        with patch('globule.core.factories.OllamaEmbeddingProvider') as mock_provider, \
             patch('globule.core.factories.OllamaEmbeddingAdapter') as mock_adapter:
            
            mock_provider_instance = Mock()
            mock_provider.return_value = mock_provider_instance
            mock_adapter_instance = Mock()
            mock_adapter.return_value = mock_adapter_instance
            
            embedding_result = EmbeddingAdapterFactory.create(config)
            
            mock_provider.assert_called_once_with(
                base_url='https://complex.example.com:11434',
                model='llama3.2:1b',
                timeout=30
            )

    def test_factory_error_messages(self):
        """Test that factory error messages are informative."""
        config = PydanticConfigManager(overrides={
            'embedding': {'provider': 'unknown_provider'}
        })
        
        with pytest.raises(ConfigError) as exc_info:
            EmbeddingAdapterFactory.create(config)
        
        error_msg = str(exc_info.value)
        assert 'unknown_provider' in error_msg
        assert 'Unsupported embedding provider' in error_msg

    def test_factory_method_coverage(self):
        """Test that all factory methods are covered."""
        config = PydanticConfigManager()
        
        # Test that all factory classes have create methods
        assert hasattr(EmbeddingAdapterFactory, 'create')
        assert hasattr(StorageManagerFactory, 'create')
        assert hasattr(ParserProviderFactory, 'create')
        assert hasattr(OrchestratorFactory, 'create')
        
        # Test that create methods are static/class methods
        assert callable(EmbeddingAdapterFactory.create)
        assert callable(StorageManagerFactory.create)
        assert callable(ParserProviderFactory.create)
        assert callable(OrchestratorFactory.create)