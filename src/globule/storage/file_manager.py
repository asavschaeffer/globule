"""
File Management for Globule.

Handles saving and loading globules as markdown files with YAML frontmatter,
implementing the UUID-based canonical ID system as specified in the refactoring plan.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from globule.core.models import ProcessedGlobule, FileDecision
from globule.config.settings import get_config


class FileManager:
    """
    Manages file operations for globules with UUID-based canonical IDs.
    
    Files are saved with human-readable names for user convenience,
    but the UUID is embedded in YAML frontmatter as the canonical identifier.
    """
    
    def __init__(self):
        self.config = get_config()
        self.base_path = self.config.get_storage_dir() / "files"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_globule_to_file(self, globule: ProcessedGlobule) -> Path:
        """
        Save a globule as a markdown file with YAML frontmatter.
        
        Args:
            globule: The processed globule to save
            
        Returns:
            Path to the saved file
        """
        # Generate human-readable filename
        filename = self._generate_filename(globule)
        
        # Determine file path
        if globule.file_decision and globule.file_decision.semantic_path:
            file_dir = self.base_path / globule.file_decision.semantic_path
            file_path = file_dir / filename
        else:
            # Default to organized by date and category
            category = globule.parsed_data.get('category', 'notes')
            date_dir = globule.created_at.strftime('%Y/%m')
            file_dir = self.base_path / category / date_dir
            file_path = file_dir / filename
        
        # Ensure directory exists
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate YAML frontmatter
        frontmatter = self._generate_frontmatter(globule)
        
        # Create markdown content
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{globule.text}\n"
        
        # Write file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def load_globule_from_file(self, file_path: Path) -> Optional[ProcessedGlobule]:
        """
        Load a globule from a markdown file using the UUID from frontmatter.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            ProcessedGlobule if successful, None if file is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML frontmatter and content
            frontmatter, text = self._parse_frontmatter(content)
            
            if not frontmatter or 'uuid' not in frontmatter:
                return None
            
            # Create ProcessedGlobule from frontmatter and content
            globule = ProcessedGlobule(
                id=frontmatter['uuid'],
                text=text.strip(),
                parsed_data=frontmatter.get('parsed_data', {}),
                parsing_confidence=frontmatter.get('parsing_confidence', 0.0),
                embedding_confidence=frontmatter.get('embedding_confidence', 0.0),
                orchestration_strategy=frontmatter.get('orchestration_strategy', 'parallel'),
                confidence_scores=frontmatter.get('confidence_scores', {}),
                processing_time_ms=frontmatter.get('processing_time_ms', {}),
                semantic_neighbors=frontmatter.get('semantic_neighbors', []),
                processing_notes=frontmatter.get('processing_notes', []),
                created_at=datetime.fromisoformat(frontmatter.get('created_at', datetime.now().isoformat())),
                modified_at=datetime.fromisoformat(frontmatter.get('modified_at', datetime.now().isoformat()))
            )
            
            # Create file decision if path info exists
            if frontmatter.get('file_decision'):
                fd_data = frontmatter['file_decision']
                globule.file_decision = FileDecision(
                    semantic_path=Path(fd_data.get('semantic_path', '')),
                    filename=fd_data.get('filename', file_path.name),
                    metadata=fd_data.get('metadata', {}),
                    confidence=fd_data.get('confidence', 0.8),
                    alternative_paths=[Path(p) for p in fd_data.get('alternative_paths', [])]
                )
            
            return globule
            
        except Exception as e:
            print(f"Error loading globule from {file_path}: {e}")
            return None
    
    def find_globule_by_uuid(self, uuid: str) -> Optional[Path]:
        """
        Find a file by its UUID, searching through all markdown files.
        
        Args:
            uuid: The canonical UUID to search for
            
        Returns:
            Path to the file if found, None otherwise
        """
        # Search through all .md files in the base directory
        for md_file in self.base_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                frontmatter, _ = self._parse_frontmatter(content)
                if frontmatter and frontmatter.get('uuid') == uuid:
                    return md_file
                    
            except Exception:
                continue  # Skip files that can't be read
        
        return None
    
    def update_file_from_globule(self, globule: ProcessedGlobule) -> Optional[Path]:
        """
        Update an existing file or create a new one based on the globule's UUID.
        
        Args:
            globule: The processed globule to save
            
        Returns:
            Path to the updated/created file
        """
        if not globule.id:
            # Generate new UUID if none exists
            import uuid
            globule.id = str(uuid.uuid4())
        
        # Try to find existing file
        existing_file = self.find_globule_by_uuid(globule.id)
        
        if existing_file:
            # Update existing file
            return self._update_existing_file(existing_file, globule)
        else:
            # Create new file
            return self.save_globule_to_file(globule)
    
    def _generate_filename(self, globule: ProcessedGlobule) -> str:
        """Generate a human-readable filename for the globule."""
        # Use title from parsed data or create from text
        if globule.parsed_data and 'title' in globule.parsed_data:
            title = globule.parsed_data['title']
        else:
            # Create title from first 50 characters
            title = globule.text[:50].strip()
        
        # Clean title for filename
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'[-\s]+', '-', clean_title).strip('-')
        
        # Limit length and add .md extension
        filename = clean_title[:50].lower() + '.md'
        
        return filename
    
    def _generate_frontmatter(self, globule: ProcessedGlobule) -> Dict[str, Any]:
        """Generate YAML frontmatter with UUID and metadata."""
        frontmatter = {
            'uuid': globule.id,
            'created_at': globule.created_at.isoformat(),
            'modified_at': globule.modified_at.isoformat(),
            'parsed_data': globule.parsed_data,
            'parsing_confidence': globule.parsing_confidence,
            'embedding_confidence': globule.embedding_confidence,
            'orchestration_strategy': globule.orchestration_strategy,
            'confidence_scores': globule.confidence_scores,
            'processing_time_ms': globule.processing_time_ms,
            'semantic_neighbors': globule.semantic_neighbors,
            'processing_notes': globule.processing_notes
        }
        
        # Add file decision if present
        if globule.file_decision:
            frontmatter['file_decision'] = {
                'semantic_path': str(globule.file_decision.semantic_path),
                'filename': globule.file_decision.filename,
                'metadata': globule.file_decision.metadata,
                'confidence': globule.file_decision.confidence,
                'alternative_paths': [str(p) for p in globule.file_decision.alternative_paths]
            }
        
        return frontmatter
    
    def _parse_frontmatter(self, content: str) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Parse YAML frontmatter from markdown content.
        
        Returns:
            (frontmatter_dict, content_text)
        """
        if not content.startswith('---'):
            return None, content
        
        try:
            # Find the closing ---
            end_marker = content.find('---', 3)
            if end_marker == -1:
                return None, content
            
            # Extract and parse YAML
            yaml_content = content[3:end_marker].strip()
            frontmatter = yaml.safe_load(yaml_content)
            
            # Extract remaining content
            text_content = content[end_marker + 3:].strip()
            
            return frontmatter, text_content
            
        except Exception:
            return None, content
    
    def _update_existing_file(self, file_path: Path, globule: ProcessedGlobule) -> Path:
        """Update an existing file with new globule data."""
        # Update the modified_at timestamp
        globule.modified_at = datetime.now()
        
        # Generate new frontmatter and content
        frontmatter = self._generate_frontmatter(globule)
        content = f"---\n{yaml.dump(frontmatter, default_flow_style=False)}---\n\n{globule.text}\n"
        
        # Write updated content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path