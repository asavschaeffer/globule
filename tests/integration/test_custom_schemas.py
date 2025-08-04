"""
Integration tests for custom schema functionality.

Test successful valet parsing with real LLM per spec.
"""

import pytest
import asyncio
import time

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
        
        # Assert extracted values match expected fields
        assert result.get('valet_name').strip('.') == 'John D'
        assert result.get('vehicle_make') == 'Toyota'
        assert result.get('vehicle_model') == 'Camry'
        assert result.get('license_plate') == '5-SAM-123'
        assert result.get('parking_spot') == 'C7'
        
        # Damage notes should contain reference to scratch
        damage_notes = result.get('damage_notes', [])
        assert len(damage_notes) > 0
        assert any('scratch' in note.lower() for note in damage_notes)
        
        # Verify result validates against valet schema (exclude metadata added by parser)
        validation_data = {k: v for k, v in result.items() if k != 'metadata'}
        is_valid, errors = schema_manager.validate_data(validation_data, 'valet')
        assert is_valid, f"Validation failed: {errors}"
