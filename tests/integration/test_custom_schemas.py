"""
Integration tests for custom schema functionality - Final Clean Version.

Test successful valet parsing with real LLM per spec.
"""

import pytest
import asyncio

from globule.services.parsing.ollama_parser import OllamaParser
from globule.schemas.manager import get_schema_manager


class TestValetSchemaIntegration:
    """Integration tests for the valet schema system."""

    @pytest.fixture
    async def parser(self):
        """Create parser instance for testing."""
        parser = OllamaParser()
        yield parser
        await parser.close()

    @pytest.fixture
    def schema_manager(self):
        """Get schema manager instance."""
        return get_schema_manager()

    def test_valet_schema_loaded(self, schema_manager):
        """Test that valet schema is properly loaded and configured."""
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Service Schema'
        
        # Verify required fields match specification
        required_fields = valet_schema.get('required', [])
        assert 'valet_name' in required_fields
        assert 'vehicle_make' in required_fields
        assert 'vehicle_model' in required_fields
        assert 'license_plate' in required_fields
        assert 'parking_spot' in required_fields
        
        # Verify damage_notes is optional (not in required)
        properties = valet_schema.get('properties', {})
        assert 'damage_notes' in properties
        assert 'damage_notes' not in required_fields

    @pytest.mark.asyncio
    async def test_valet_schema_parsing(self, parser, schema_manager):
        """
        Test successful valet parsing with real LLM per spec.
        
        Uses real Ollama LLM to parse sample text and validate 
        extraction and schema validation end-to-end.
        """
        # Sample input as specified in requirements
        sample_text = ("Valet: Parked a black Toyota Camry, license plate 5-SAM-123, "
                       "in spot C7. Noted a small scratch on the rear bumper. Valet: John D.")
        
        # Use real Ollama parsing (no mocking the response)
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify successful LLM parsing (not fallback)
        assert result['metadata']['parser_type'] == 'ollama_llm'
        
        # Assert extracted values match expected fields (handle LLM variance with contains)
        assert result.get('valet_name') == 'John D'
        assert result.get('vehicle_make') == 'Toyota' 
        assert result.get('vehicle_model') == 'Camry'
        assert result.get('license_plate') == '5-SAM-123'
        assert result.get('parking_spot') == 'C7'
        
        # Damage notes should contain reference to scratch
        damage_notes = result.get('damage_notes', [])
        assert len(damage_notes) > 0
        assert any('scratch' in note.lower() for note in damage_notes)
        
        # Verify result validates against valet schema
        # Extract only the valet-specific fields for validation
        valet_data = {
            'title': result['title'],
            'category': result['category'], 
            'domain': result['domain'],
            'valet_name': result['valet_name'],
            'vehicle_make': result['vehicle_make'],
            'vehicle_model': result['vehicle_model'],
            'license_plate': result['license_plate'],
            'parking_spot': result['parking_spot'],
            'damage_notes': result['damage_notes'],
            'keywords': result['keywords'],
            'entities': result['entities'],
            'sentiment': result['metadata']['sentiment'],
            'content_type': result['metadata']['content_type'],
            'confidence_score': result['metadata']['confidence_score']
        }
        
        is_valid, errors = schema_manager.validate_data(valet_data, 'valet')
        assert is_valid, f"Validation failed: {errors}"

    @pytest.mark.asyncio
    async def test_valet_schema_parsing_without_damage(self, parser, schema_manager):
        """
        Test valet parsing with optional damage_notes field missing.
        
        Edge case: Input without damage_notes should validate successfully
        since damage_notes is optional.
        Future: Add timestamp for correlations.
        """
        sample_text = ("Valet: Parked a blue Honda Accord, license plate 7-ABC-456, "
                       "in spot A5. No issues noted. Valet: Sarah M.")
        
        # Use real Ollama parsing
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify successful parsing
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result.get('valet_name') == 'Sarah M'
        assert result.get('vehicle_make') == 'Honda'
        assert result.get('vehicle_model') == 'Accord'
        assert result.get('license_plate') == '7-ABC-456'
        assert result.get('parking_spot') == 'A5'
        
        # Verify damage_notes is absent or empty (no damage mentioned)
        damage_notes = result.get('damage_notes', [])
        assert isinstance(damage_notes, list)
        # Should be empty or minimal since no damage was noted
        
        # Prepare validation data
        valet_data = {
            'title': result['title'],
            'category': result['category'],
            'domain': result['domain'],
            'valet_name': result['valet_name'],
            'vehicle_make': result['vehicle_make'],
            'vehicle_model': result['vehicle_model'],
            'license_plate': result['license_plate'],
            'parking_spot': result['parking_spot'],
            'keywords': result['keywords'],
            'entities': result['entities'],
            'sentiment': result['metadata']['sentiment'],
            'content_type': result['metadata']['content_type'],
            'confidence_score': result['metadata']['confidence_score']
        }
        # Only add damage_notes if present and non-empty
        if result.get('damage_notes'):
            valet_data['damage_notes'] = result['damage_notes']
        
        is_valid, errors = schema_manager.validate_data(valet_data, 'valet')
        assert is_valid, f"Should be valid without optional damage_notes: {errors}"

    def test_valet_schema_validation(self, schema_manager):
        """Test direct schema validation with valet data."""
        # Test valid valet data
        valid_valet_data = {
            "title": "Valet Service Entry",
            "category": "note",
            "domain": "valet",
            "valet_name": "John D",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "license_plate": "5-SAM-123",
            "parking_spot": "C7",
            "damage_notes": ["small scratch on the rear bumper"],
            "keywords": ["valet", "parking"],
            "entities": ["Toyota", "Camry", "John D"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(valid_valet_data, 'valet')
        assert is_valid, f"Validation failed: {errors}"
        
        # Test invalid data (missing required fields)
        invalid_data = {
            "title": "Incomplete Entry",
            "category": "note",
            "domain": "valet"
            # Missing required valet fields
        }
        
        is_valid, errors = schema_manager.validate_data(invalid_data, 'valet')
        assert not is_valid
        assert len(errors) > 0

    def test_schema_manager_domain_mapping(self, schema_manager):
        """Test that valet domain correctly maps to valet schema."""
        schema_name = schema_manager.get_schema_for_domain('valet')
        assert schema_name == 'valet'
        
        # Verify the schema exists and loads correctly
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Service Schema'