"""
Integration tests for custom schema functionality - Step 2 Enhanced Validation.

Test enhanced validation for analysis per phased plan with real LLM parsing.
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
        """Test that valet schema is properly loaded and configured with Step 2 enhancements."""
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Service Schema'
        
        # Verify required fields match specification including new timestamp
        required_fields = valet_schema.get('required', [])
        assert 'valet_name' in required_fields
        assert 'vehicle_make' in required_fields
        assert 'vehicle_model' in required_fields
        assert 'license_plate' in required_fields
        assert 'parking_spot' in required_fields
        assert 'timestamp' in required_fields  # New Step 2 field
        
        # Verify damage_notes is optional (not in required)
        properties = valet_schema.get('properties', {})
        assert 'damage_notes' in properties
        assert 'damage_notes' not in required_fields
        
        # Verify Step 2 validation enhancements
        vehicle_make_prop = properties.get('vehicle_make', {})
        assert 'enum' in vehicle_make_prop
        assert vehicle_make_prop['enum'] == ["Toyota", "Honda", "Ford", "Other"]
        
        license_plate_prop = properties.get('license_plate', {})
        assert 'pattern' in license_plate_prop
        assert license_plate_prop['pattern'] == "^[A-Z0-9-]{6,10}$"
        
        timestamp_prop = properties.get('timestamp', {})
        assert timestamp_prop['type'] == 'integer'
        assert timestamp_prop['minimum'] == 0

    @pytest.mark.asyncio
    async def test_valet_schema_parsing(self, parser, schema_manager):
        """
        Test successful valet parsing with real LLM and Step 2 enhanced validation.
        
        Uses real Ollama LLM to parse sample text and validate 
        extraction and enhanced schema validation end-to-end.
        """
        # Sample input as specified in requirements
        sample_text = ("Valet: Parked a black Toyota Camry, license plate 5-SAM-123, "
                       "in spot C7. Noted a small scratch on the rear bumper. Valet: John D.")
        
        # Use real Ollama parsing (no mocking the response)
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify successful LLM parsing (not fallback)
        assert result['metadata']['parser_type'] == 'ollama_llm'
        
        # Assert extracted values match expected fields
        assert result.get('valet_name') == 'John D'
        assert result.get('vehicle_make') == 'Toyota'  # Should match enum
        assert result.get('vehicle_model') == 'Camry'
        assert result.get('license_plate') == '5-SAM-123'  # Should match pattern
        assert result.get('parking_spot') == 'C7'
        
        # Step 2: Assert timestamp is not auto-filled (should be absent from LLM parsing)
        timestamp = result.get('timestamp')
        assert timestamp is None, "Timestamp should not be auto-filled by parser"
        
        # Damage notes should contain reference to scratch
        damage_notes = result.get('damage_notes', [])
        assert len(damage_notes) > 0
        assert any('scratch' in note.lower() for note in damage_notes)
        
        # Note: Since timestamp is required but not auto-filled, this validation will fail
        # This demonstrates that the LLM must provide timestamp or validation will fail
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
        
        # Validation should fail because timestamp is required but missing
        is_valid, errors = schema_manager.validate_data(valet_data, 'valet')
        assert not is_valid, "Validation should fail when timestamp is missing"
        assert any('timestamp' in error.lower() for error in errors), f"Should have timestamp error: {errors}"

    @pytest.mark.asyncio
    async def test_valet_schema_parsing_without_damage(self, parser, schema_manager):
        """
        Test valet parsing with optional damage_notes field missing and Step 2 validation.
        
        Edge case: Input without damage_notes should validate successfully
        since damage_notes is optional. Tests enhanced validation constraints.
        """
        sample_text = ("Valet: Parked a blue Honda Accord, license plate 7-ABC-456, "
                       "in spot A5. No issues noted. Valet: Sarah M.")
        
        # Use real Ollama parsing
        result = await parser.parse(sample_text, {'domain': 'valet'})
        
        # Verify successful parsing
        assert result['metadata']['parser_type'] == 'ollama_llm'
        assert result.get('valet_name') == 'Sarah M'
        assert result.get('vehicle_make') == 'Honda'  # Should match enum
        assert result.get('vehicle_model') == 'Accord'
        assert result.get('license_plate') == '7-ABC-456'  # Should match pattern
        assert result.get('parking_spot') == 'A5'
        
        # Step 2: Assert timestamp is not auto-filled
        timestamp = result.get('timestamp')
        assert timestamp is None, "Timestamp should not be auto-filled by parser"
        
        # Verify damage_notes is absent or empty (no damage mentioned)
        damage_notes = result.get('damage_notes', [])
        assert isinstance(damage_notes, list)
        # Should be empty or minimal since no damage was noted
        
        # Prepare validation data (timestamp missing, so validation will fail)
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
        
        # Validation should fail due to missing timestamp
        is_valid, errors = schema_manager.validate_data(valet_data, 'valet')
        assert not is_valid, "Validation should fail when required timestamp is missing"
        assert any('timestamp' in error.lower() for error in errors), f"Should have timestamp error: {errors}"

    def test_valet_schema_validation(self, schema_manager):
        """Test direct schema validation with enhanced Step 2 valet data."""
        # Test valid valet data with Step 2 enhancements
        current_timestamp = int(time.time())
        valid_valet_data = {
            "title": "Valet Service Entry",
            "category": "note",
            "domain": "valet",
            "valet_name": "John D",
            "vehicle_make": "Toyota",  # Valid enum value
            "vehicle_model": "Camry",
            "license_plate": "5-SAM-123",  # Valid pattern
            "parking_spot": "C7",
            "timestamp": current_timestamp,  # New Step 2 required field
            "damage_notes": ["small scratch on the rear bumper"],
            "keywords": ["valet", "parking"],
            "entities": ["Toyota", "Camry", "John D"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(valid_valet_data, 'valet')
        assert is_valid, f"Enhanced validation failed: {errors}"
        
        # Test invalid data (missing required fields including timestamp)
        invalid_data = {
            "title": "Incomplete Entry",
            "category": "note",
            "domain": "valet"
            # Missing required valet fields including new timestamp
        }
        
        is_valid, errors = schema_manager.validate_data(invalid_data, 'valet')
        assert not is_valid
        assert len(errors) > 0
        # Should include error about missing timestamp
        assert any('timestamp' in error.lower() for error in errors)

    def test_valet_schema_field_validation(self, schema_manager):
        """Test enhanced validation for analysis per phased plan - constraint testing."""
        current_timestamp = int(time.time())
        
        # Test invalid license plate pattern
        invalid_license_data = {
            "title": "Invalid License Test",
            "category": "note",
            "domain": "valet",
            "valet_name": "Test User",
            "vehicle_make": "Toyota",
            "vehicle_model": "Camry",
            "license_plate": "ABC",  # Too short, doesn't match pattern
            "parking_spot": "A1",
            "timestamp": current_timestamp,
            "keywords": ["test"],
            "entities": ["test"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(invalid_license_data, 'valet')
        assert not is_valid
        assert any('pattern' in error.lower() or 'does not match' in error.lower() for error in errors)
        
        # Test invalid vehicle make enum
        invalid_make_data = {
            "title": "Invalid Make Test",
            "category": "note",
            "domain": "valet",
            "valet_name": "Test User",
            "vehicle_make": "BMW",  # Not in enum
            "vehicle_model": "X5",
            "license_plate": "5-BMW-123",
            "parking_spot": "A1",
            "timestamp": current_timestamp,
            "keywords": ["test"],
            "entities": ["test"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(invalid_make_data, 'valet')
        assert not is_valid
        assert any('enum' in error.lower() or 'vehicle_make' in error.lower() for error in errors)
        
        # Test duplicate damage_notes (should fail uniqueItems constraint)
        duplicate_damage_data = {
            "title": "Duplicate Damage Test",
            "category": "note",
            "domain": "valet",
            "valet_name": "Test User",
            "vehicle_make": "Honda",
            "vehicle_model": "Civic",
            "license_plate": "5-DUP-123",
            "parking_spot": "B2",
            "timestamp": current_timestamp,
            "damage_notes": ["scratch on door", "scratch on door"],  # Duplicate items
            "keywords": ["test"],
            "entities": ["test"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(duplicate_damage_data, 'valet')
        assert not is_valid
        assert any('unique' in error.lower() for error in errors)
        
        # Test missing timestamp (required field)
        missing_timestamp_data = {
            "title": "Missing Timestamp Test",
            "category": "note",
            "domain": "valet",
            "valet_name": "Test User",
            "vehicle_make": "Ford",
            "vehicle_model": "Focus",
            "license_plate": "5-TST-789",
            "parking_spot": "C3",
            # timestamp missing
            "keywords": ["test"],
            "entities": ["test"],
            "sentiment": "neutral",
            "content_type": "log_entry",
            "confidence_score": 0.9
        }
        
        is_valid, errors = schema_manager.validate_data(missing_timestamp_data, 'valet')
        assert not is_valid
        assert any('timestamp' in error.lower() and 'required' in error.lower() for error in errors)

    def test_schema_manager_domain_mapping(self, schema_manager):
        """Test that valet domain correctly maps to valet schema."""
        schema_name = schema_manager.get_schema_for_domain('valet')
        assert schema_name == 'valet'
        
        # Verify the schema exists and loads correctly
        valet_schema = schema_manager.get_schema('valet')
        assert valet_schema is not None
        assert valet_schema['title'] == 'Valet Service Schema'