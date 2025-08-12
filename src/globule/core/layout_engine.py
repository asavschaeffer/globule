"""
Layout Engine: Interprets schema canvas configurations for consistent rendering.

This module provides the core logic for positioning and styling modules based on
their schema canvas_config definitions. It ensures consistent layouts across
different frontends (TUI, Web) while allowing for frontend-specific customization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from globule.schemas.manager import SchemaManager

logger = logging.getLogger(__name__)


class LayoutType(Enum):
    """Available layout types for modules."""
    WIDGET = "widget"
    PANEL = "panel"
    FULLSCREEN = "fullscreen"
    SIDEBAR = "sidebar"


class Position(Enum):
    """Available positioning options."""
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"  
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"
    SIDEBAR_LEFT = "sidebar-left"
    SIDEBAR_RIGHT = "sidebar-right"
    FULL_WIDTH = "full-width"


class Size(Enum):
    """Available size options."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    AUTO = "auto"
    FULL = "full"


@dataclass
class LayoutConfig:
    """Parsed layout configuration."""
    layout_type: LayoutType
    position: Position
    size: Size
    css_classes: List[str]
    tui_style: Dict[str, Any]
    web_style: Dict[str, Any]
    schema_name: str


@dataclass
class CanvasModule:
    """Represents a module positioned on the canvas."""
    id: str
    name: str
    content: str
    layout: LayoutConfig
    metadata: Dict[str, Any]


class LayoutEngine:
    """
    Core layout engine that interprets schema configurations and positions modules.
    
    This engine provides a unified way to handle layout positioning that works
    across different frontend implementations.
    """
    
    def __init__(self, schema_manager: Optional[SchemaManager] = None):
        """
        Initialize layout engine.
        
        Args:
            schema_manager: Schema manager instance for loading canvas configs
        """
        self.schema_manager = schema_manager or SchemaManager()
        self._layout_cache: Dict[str, LayoutConfig] = {}
        self._grid_assignments: Dict[Position, List[str]] = {}
        
        # Initialize grid
        self._initialize_grid()
    
    def _initialize_grid(self):
        """Initialize the grid layout structure."""
        for position in Position:
            self._grid_assignments[position] = []
    
    def get_layout_config(self, schema_name: str) -> Optional[LayoutConfig]:
        """
        Get parsed layout configuration for a schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            LayoutConfig instance or None if no canvas config found
        """
        if schema_name in self._layout_cache:
            return self._layout_cache[schema_name]
        
        canvas_config = self.schema_manager.get_canvas_config(schema_name)
        if not canvas_config:
            return None
        
        try:
            layout_config = self._parse_canvas_config(schema_name, canvas_config)
            self._layout_cache[schema_name] = layout_config
            return layout_config
            
        except Exception as e:
            logger.error(f"Failed to parse canvas config for {schema_name}: {e}")
            return None
    
    def _parse_canvas_config(self, schema_name: str, canvas_config: Dict[str, Any]) -> LayoutConfig:
        """Parse canvas configuration into LayoutConfig object."""
        layout = canvas_config.get('layout', {})
        
        return LayoutConfig(
            layout_type=LayoutType(layout.get('type', 'widget')),
            position=Position(layout.get('position', 'center')),
            size=Size(layout.get('size', 'medium')),
            css_classes=layout.get('css_classes', []),
            tui_style=canvas_config.get('tui_style', {}),
            web_style=canvas_config.get('web_style', {}),
            schema_name=schema_name
        )
    
    def create_canvas_module(self, id: str, name: str, content: str, 
                           schema_name: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[CanvasModule]:
        """
        Create a canvas module with layout configuration.
        
        Args:
            id: Unique identifier for the module
            name: Display name for the module
            content: Module content
            schema_name: Schema name to get layout config from
            metadata: Optional metadata dictionary
            
        Returns:
            CanvasModule instance or None if schema has no layout config
        """
        layout_config = self.get_layout_config(schema_name)
        if not layout_config:
            logger.debug(f"No layout config found for schema: {schema_name}")
            return None
        
        return CanvasModule(
            id=id,
            name=name,
            content=content,
            layout=layout_config,
            metadata=metadata or {}
        )
    
    def position_modules(self, modules: List[CanvasModule]) -> Dict[Position, List[CanvasModule]]:
        """
        Organize modules by their position configuration.
        
        Args:
            modules: List of canvas modules to position
            
        Returns:
            Dictionary mapping positions to lists of modules
        """
        positioned = {}
        for position in Position:
            positioned[position] = []
        
        for module in modules:
            position = module.layout.position
            positioned[position].append(module)
        
        return positioned
    
    def generate_tui_layout(self, modules: List[CanvasModule]) -> Dict[str, Any]:
        """
        Generate TUI-specific layout configuration.
        
        Args:
            modules: List of canvas modules
            
        Returns:
            Dictionary with TUI layout configuration
        """
        positioned = self.position_modules(modules)
        
        # Create grid layout for Textual
        layout_config = {
            "type": "grid",
            "grid_size": [3, 3],  # 3x3 grid
            "grid_gutter": [1, 1],
            "areas": {},
            "widgets": {}
        }
        
        # Map positions to grid areas
        position_mapping = {
            Position.TOP_LEFT: {"column": "1", "row": "1"},
            Position.TOP_CENTER: {"column": "2", "row": "1"}, 
            Position.TOP_RIGHT: {"column": "3", "row": "1"},
            Position.CENTER_LEFT: {"column": "1", "row": "2"},
            Position.CENTER: {"column": "2", "row": "2"},
            Position.CENTER_RIGHT: {"column": "3", "row": "2"},
            Position.BOTTOM_LEFT: {"column": "1", "row": "3"},
            Position.BOTTOM_CENTER: {"column": "2", "row": "3"},
            Position.BOTTOM_RIGHT: {"column": "3", "row": "3"},
            Position.FULL_WIDTH: {"column": "1 / 4", "row": "2"}
        }
        
        for position, module_list in positioned.items():
            if not module_list:
                continue
                
            grid_pos = position_mapping.get(position)
            if not grid_pos:
                continue
            
            for i, module in enumerate(module_list):
                widget_id = f"{module.id}_{i}"
                
                # Add widget configuration
                layout_config["widgets"][widget_id] = {
                    "module": module,
                    "style": module.layout.tui_style,
                    "grid_column": grid_pos["column"],
                    "grid_row": grid_pos["row"]
                }
        
        return layout_config
    
    def generate_web_layout(self, modules: List[CanvasModule]) -> Dict[str, Any]:
        """
        Generate Web-specific layout configuration.
        
        Args:
            modules: List of canvas modules
            
        Returns:
            Dictionary with Web layout configuration
        """
        positioned = self.position_modules(modules)
        
        layout_config = {
            "type": "grid",
            "css_grid_template": "1fr 2fr 1fr / 1fr 2fr 1fr",
            "grid_areas": {
                "top-left": "1 / 1",
                "top-center": "1 / 2",
                "top-right": "1 / 3",
                "center-left": "2 / 1", 
                "center": "2 / 2",
                "center-right": "2 / 3",
                "bottom-left": "3 / 1",
                "bottom-center": "3 / 2",
                "bottom-right": "3 / 3",
                "full-width": "2 / 1 / 2 / 4"
            },
            "components": {}
        }
        
        for position, module_list in positioned.items():
            if not module_list:
                continue
                
            for i, module in enumerate(module_list):
                component_id = f"{module.id}_{i}"
                
                layout_config["components"][component_id] = {
                    "module": module,
                    "style": module.layout.web_style,
                    "grid_area": layout_config["grid_areas"].get(position.value, "center")
                }
        
        return layout_config
    
    def get_available_positions(self) -> List[str]:
        """Get list of all available position names."""
        return [pos.value for pos in Position]
    
    def get_available_sizes(self) -> List[str]:
        """Get list of all available size names."""
        return [size.value for size in Size]
    
    def get_available_layout_types(self) -> List[str]:
        """Get list of all available layout type names."""
        return [layout_type.value for layout_type in LayoutType]
    
    def validate_layout_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a layout configuration.
        
        Args:
            config: Layout configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if 'layout' not in config:
            errors.append("Missing 'layout' section")
            return False, errors
        
        layout = config['layout']
        
        # Check required fields
        required = ['type', 'position', 'size']
        for field in required:
            if field not in layout:
                errors.append(f"Missing required field: layout.{field}")
        
        # Validate enum values
        if 'type' in layout:
            try:
                LayoutType(layout['type'])
            except ValueError:
                errors.append(f"Invalid layout type: {layout['type']}")
        
        if 'position' in layout:
            try:
                Position(layout['position'])
            except ValueError:
                errors.append(f"Invalid position: {layout['position']}")
        
        if 'size' in layout:
            try:
                Size(layout['size'])
            except ValueError:
                errors.append(f"Invalid size: {layout['size']}")
        
        return len(errors) == 0, errors