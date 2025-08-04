"""
Schema management for dynamic loading and validation.

This module provides utilities for loading, caching, and validating
JSON schemas used throughout the Globule parsing system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from functools import lru_cache

import jsonschema
from jsonschema import Draft7Validator

logger = logging.getLogger(__name__)


class SchemaManager:
    """Manages JSON schemas for content parsing and validation."""
    
    def __init__(self):
        """Initialize schema manager with built-in schemas."""
        self.schemas_dir = Path(__file__).parent
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validator_cache: Dict[str, Draft7Validator] = {}
        
        # Preload built-in schemas
        self._load_builtin_schemas()
    
    def _load_builtin_schemas(self) -> None:
        """Load all built-in schemas from the schemas directory."""
        schema_files = list(self.schemas_dir.glob("*.json"))
        
        for schema_file in schema_files:
            schema_name = schema_file.stem
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                self._schema_cache[schema_name] = schema_data
                self._validator_cache[schema_name] = Draft7Validator(schema_data)
                
                logger.debug(f"Loaded schema: {schema_name}")
                
            except Exception as e:
                logger.error(f"Failed to load schema {schema_name}: {e}")
    
    def get_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a schema by name.
        
        Args:
            schema_name: Name of the schema (e.g., 'default', 'academic', 'technical')
            
        Returns:
            Schema dictionary or None if not found
        """
        return self._schema_cache.get(schema_name)
    
    def get_validator(self, schema_name: str) -> Optional[Draft7Validator]:
        """
        Get a validator for a schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Validator instance or None if schema not found
        """
        return self._validator_cache.get(schema_name)
    
    def validate_data(self, data: Dict[str, Any], schema_name: str) -> tuple[bool, List[str]]:
        """
        Validate data against a schema.
        
        Args:
            data: Data to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        validator = self.get_validator(schema_name)
        if not validator:
            return False, [f"Schema '{schema_name}' not found"]
        
        errors = []
        try:
            # Collect all validation errors
            for error in validator.iter_errors(data):
                # Format error message with path context
                if error.absolute_path:
                    path = " -> ".join(str(p) for p in error.absolute_path)
                    errors.append(f"{path}: {error.message}")
                else:
                    errors.append(error.message)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]
    
    def get_available_schemas(self) -> List[str]:
        """Get list of available schema names."""
        return list(self._schema_cache.keys())
    
    def get_schema_for_domain(self, domain: str) -> str:
        """
        Get the best schema name for a given domain.
        
        Args:
            domain: Domain name (e.g., 'technical', 'creative', 'academic')
            
        Returns:
            Schema name to use for this domain
        """
        # Map domains to schema names
        domain_mappings = {
            'technical': 'technical',
            'creative': 'creative', 
            'academic': 'academic',
            'valet': 'valet',
            'computer_science': 'academic',
            'research': 'academic',
            'engineering': 'technical',
            'software': 'technical',
            'art': 'creative',
            'writing': 'creative',
            'fiction': 'creative'
        }
        
        return domain_mappings.get(domain.lower(), 'default')
    
    def format_schema_for_llm(self, schema_name: str) -> str:
        """
        Format a schema for inclusion in LLM prompts.
        
        Args:
            schema_name: Name of schema to format
            
        Returns:
            Human-readable schema description for LLM prompts
        """
        schema = self.get_schema(schema_name)
        if not schema:
            return "No schema available"
        
        # Extract key information for LLM prompts
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        
        formatted_parts = []
        formatted_parts.append(f"Schema: {schema.get('title', schema_name)}")
        
        if 'description' in schema:
            formatted_parts.append(f"Description: {schema['description']}")
        
        formatted_parts.append("Required fields:")
        
        for field in required:
            if field in properties:
                prop = properties[field]
                field_desc = f"  - {field}: {prop.get('description', 'No description')}"
                
                # Add type information
                if 'type' in prop:
                    field_desc += f" (type: {prop['type']})"
                
                # Add enum values if present
                if 'enum' in prop:
                    field_desc += f" [options: {', '.join(prop['enum'])}]"
                elif prop.get('type') == 'array' and 'items' in prop and 'enum' in prop['items']:
                    field_desc += f" [item options: {', '.join(prop['items']['enum'])}]"
                
                formatted_parts.append(field_desc)
        
        # Add optional fields if any
        optional_fields = [f for f in properties if f not in required]
        if optional_fields:
            formatted_parts.append("Optional fields:")
            for field in optional_fields[:5]:  # Limit to first 5 optional fields
                if field in properties:
                    prop = properties[field]
                    field_desc = f"  - {field}: {prop.get('description', 'No description')}"
                    formatted_parts.append(field_desc)
        
        return "\n".join(formatted_parts)
    
    def load_custom_schema(self, schema_path: Path, schema_name: str) -> bool:
        """
        Load a custom schema from a file path.
        
        Args:
            schema_path: Path to JSON schema file
            schema_name: Name to register the schema under
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = json.load(f)
            
            # Validate that it's a valid JSON schema
            Draft7Validator.check_schema(schema_data)
            
            self._schema_cache[schema_name] = schema_data
            self._validator_cache[schema_name] = Draft7Validator(schema_data)
            
            logger.info(f"Loaded custom schema: {schema_name} from {schema_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load custom schema {schema_name} from {schema_path}: {e}")
            return False


# Global schema manager instance
_schema_manager: Optional[SchemaManager] = None


def get_schema_manager() -> SchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager


# Convenience functions for common operations
def get_schema(schema_name: str) -> Optional[Dict[str, Any]]:
    """Get a schema by name."""
    return get_schema_manager().get_schema(schema_name)


def validate_data(data: Dict[str, Any], schema_name: str) -> tuple[bool, List[str]]:
    """Validate data against a schema."""
    return get_schema_manager().validate_data(data, schema_name)


def format_schema_for_llm(schema_name: str) -> str:
    """Format a schema for LLM prompts."""
    return get_schema_manager().format_schema_for_llm(schema_name)


def get_schema_for_domain(domain: str) -> str:
    """Get the best schema for a domain."""
    return get_schema_manager().get_schema_for_domain(domain)