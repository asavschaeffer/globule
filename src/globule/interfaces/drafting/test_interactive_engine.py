"""
Unit tests for InteractiveDraftingEngine.

Tests robust error handling, terminal management, and edge cases
to ensure production-ready reliability.
"""

import pytest
import asyncio
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from interactive_engine import (
    InteractiveDraftingEngine, 
    DraftingState, 
    DraftingView
)
from globule.core.models import ProcessedGlobule
from globule.clustering.semantic_clustering import SemanticCluster


class TestDraftingState:
    """Test DraftingState validation and error handling."""
    
    def test_empty_clusters_handling(self):
        """Test state handles empty clusters gracefully."""
        state = DraftingState()
        state.clusters = []
        
        cluster = state.get_current_cluster()
        assert cluster is None
        assert "No clusters available" in state.status_message
    
    def test_invalid_cluster_index(self):
        """Test state handles invalid cluster indices."""
        state = DraftingState()
        state.clusters = [Mock(id="1", label="Test")]
        state.selected_cluster_index = 5  # Invalid index
        
        cluster = state.get_current_cluster()
        assert cluster is None
        assert "Invalid cluster selection" in state.status_message
    
    def test_missing_globules_mapping(self):
        """Test state handles missing globules mapping gracefully."""
        state = DraftingState()
        mock_cluster = Mock(id="missing_cluster")
        state.clusters = [mock_cluster]
        state.globules_by_cluster = {}  # Empty mapping
        state.selected_cluster_index = 0
        
        globules = state.get_current_cluster_globules()
        assert globules == []
        assert "No globules mapped" in state.status_message
    
    def test_invalid_globule_index(self):
        """Test state handles invalid globule indices."""
        state = DraftingState()
        mock_cluster = Mock(id="cluster1")
        mock_globule = Mock(text="Test globule")
        
        state.clusters = [mock_cluster]
        state.globules_by_cluster = {"cluster1": [mock_globule]}
        state.selected_cluster_index = 0
        state.selected_globule_index = 10  # Invalid index
        
        globule = state.get_current_globule()
        assert globule is None
        assert "Invalid globule selection" in state.status_message
    
    def test_draft_text_generation(self):
        """Test draft text generation with various content."""
        state = DraftingState()
        state.topic = "Test Topic"
        state.draft_content = ["First item", "Second item"]
        
        draft_text = state.get_draft_text()
        assert "Test Topic" in draft_text
        assert "First item" in draft_text
        assert "Second item" in draft_text
        assert "Items: 2" in draft_text


class TestInteractiveDraftingEngine:
    """Test InteractiveDraftingEngine robustness and error handling."""
    
    def test_input_validation_invalid_clusters(self):
        """Test input validation rejects invalid cluster types."""
        engine = InteractiveDraftingEngine()
        
        with pytest.raises(ValueError, match="clusters must be a list"):
            asyncio.run(engine.run_interactive_session(
                "topic", "not_a_list", {}, []
            ))
    
    def test_input_validation_invalid_globules_mapping(self):
        """Test input validation rejects invalid globules mapping."""
        engine = InteractiveDraftingEngine()
        
        with pytest.raises(ValueError, match="globules_by_cluster must be a dictionary"):
            asyncio.run(engine.run_interactive_session(
                "topic", [], "not_a_dict", []
            ))
    
    def test_input_validation_invalid_all_globules(self):
        """Test input validation rejects invalid all_globules."""
        engine = InteractiveDraftingEngine()
        
        with pytest.raises(ValueError, match="all_globules must be a list"):
            asyncio.run(engine.run_interactive_session(
                "topic", [], {}, "not_a_list"
            ))
    
    @patch('sys.platform', 'linux')
    def test_terminal_setup_success_unix(self):
        """Test successful terminal setup on Unix systems."""
        engine = InteractiveDraftingEngine()
        
        with patch('termios.tcgetattr') as mock_tcgetattr, \\
             patch('tty.setraw') as mock_setraw:
            mock_tcgetattr.return_value = "mock_settings"
            
            result = engine._setup_terminal()
            assert result is True
            assert engine._old_settings == "mock_settings"
            mock_tcgetattr.assert_called_once()
            mock_setraw.assert_called_once()
    
    @patch('sys.platform', 'linux')
    def test_terminal_setup_failure_unix(self):
        """Test terminal setup failure handling on Unix systems."""
        engine = InteractiveDraftingEngine()
        
        with patch('termios.tcgetattr', side_effect=OSError("No terminal")):
            result = engine._setup_terminal()
            assert result is False
            assert "Terminal setup failed" in engine.state.status_message
    
    @patch('sys.platform', 'win32')
    def test_terminal_setup_windows(self):
        """Test terminal setup on Windows (should always succeed)."""
        engine = InteractiveDraftingEngine()
        
        result = engine._setup_terminal()
        assert result is True
    
    @patch('sys.platform', 'linux')
    def test_terminal_restore_success(self):
        """Test successful terminal restoration."""
        engine = InteractiveDraftingEngine()
        engine._old_settings = "mock_settings"
        
        with patch('termios.tcsetattr') as mock_tcsetattr:
            engine._restore_terminal()
            mock_tcsetattr.assert_called_once()
            assert engine._old_settings is None  # Should be cleared
    
    @patch('sys.platform', 'linux')
    def test_terminal_restore_failure_silent(self):
        """Test terminal restoration fails silently."""
        engine = InteractiveDraftingEngine()
        engine._old_settings = "mock_settings"
        
        with patch('termios.tcsetattr', side_effect=OSError("Terminal error")):
            # Should not raise an exception
            engine._restore_terminal()
    
    @patch('sys.platform', 'win32')
    def test_keypress_handling_windows(self):
        """Test keypress handling on Windows."""
        engine = InteractiveDraftingEngine()
        
        with patch('msvcrt.getch') as mock_getch:
            # Test normal key
            mock_getch.return_value = b'a'
            result = engine._get_keypress()
            assert result == 'a'
            
            # Test special key (arrow)
            mock_getch.side_effect = [b'\\xe0', b'H']  # Up arrow
            result = engine._get_keypress()
            assert result == 'up'
    
    @patch('sys.platform', 'linux')
    def test_keypress_handling_unix(self):
        """Test keypress handling on Unix."""
        engine = InteractiveDraftingEngine()
        
        with patch('sys.stdin.read') as mock_read:
            # Test normal key
            mock_read.return_value = 'a'
            result = engine._get_keypress()
            assert result == 'a'
            
            # Test escape sequence (arrow key)
            mock_read.side_effect = ['\\x1b', '[A']
            result = engine._get_keypress()
            assert result == 'up'
    
    def test_keypress_error_handling_windows(self):
        """Test keypress error handling on Windows."""
        engine = InteractiveDraftingEngine()
        
        with patch('sys.platform', 'win32'), \\
             patch('msvcrt.getch', side_effect=Exception("Mock error")):
            result = engine._get_keypress()
            assert result == ''
            assert "Keypress error on Windows" in engine.state.status_message
    
    def test_keypress_error_handling_unix(self):
        """Test keypress error handling on Unix."""
        engine = InteractiveDraftingEngine()
        
        with patch('sys.platform', 'linux'), \\
             patch('sys.stdin.read', side_effect=Exception("Mock error")):
            result = engine._get_keypress()
            assert result == ''
            assert "Keypress error on Unix" in engine.state.status_message
    
    @pytest.mark.asyncio
    async def test_terminal_restore_in_finally_block(self):
        """Test that terminal is restored even when exceptions occur."""
        engine = InteractiveDraftingEngine()
        engine._setup_terminal = Mock(return_value=True)
        engine._restore_terminal = Mock()
        engine._render_ui = Mock(side_effect=RuntimeError("Simulated crash"))
        engine._get_keypress = Mock(return_value='q')
        
        # Create minimal valid inputs
        mock_cluster = Mock(id="1", label="Test")
        mock_globule = Mock(text="Test")
        
        try:
            await engine.run_interactive_session(
                "test", [mock_cluster], {"1": [mock_globule]}, [mock_globule]
            )
        except:
            pass  # We expect an exception
        
        # Terminal restore should still be called
        engine._restore_terminal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_keyboard_interrupt_handling(self):
        """Test graceful handling of Ctrl+C."""
        engine = InteractiveDraftingEngine()
        engine._setup_terminal = Mock(return_value=True)
        engine._restore_terminal = Mock()
        engine._render_ui = Mock()
        engine._get_keypress = Mock(side_effect=KeyboardInterrupt())
        
        mock_cluster = Mock(id="1", label="Test")
        mock_globule = Mock(text="Test")
        
        result = await engine.run_interactive_session(
            "test", [mock_cluster], {"1": [mock_globule]}, [mock_globule]
        )
        
        assert "interrupted by user" in engine.state.status_message
        assert engine.state.should_quit is True
        engine._restore_terminal.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_clusters_fallback(self):
        """Test fallback behavior when no clusters provided."""
        engine = InteractiveDraftingEngine()
        
        result = await engine.run_interactive_session("test", [], {}, [])
        
        assert "No semantic clusters found" in result
        assert "test" in result  # Topic should be in result
    
    @pytest.mark.asyncio
    async def test_terminal_setup_failure_fallback(self):
        """Test fallback when terminal setup fails."""
        engine = InteractiveDraftingEngine()
        engine._setup_terminal = Mock(return_value=False)
        
        result = await engine.run_interactive_session("test", [], {}, [])
        
        assert "No semantic clusters found" in result
        engine._setup_terminal.assert_called_once()
    
    def test_terminal_size_validation(self):
        """Test UI handles small terminal sizes gracefully."""
        engine = InteractiveDraftingEngine()
        engine.console.width = 50  # Too small
        engine.console.height = 10  # Too small
        
        with patch.object(engine.console, 'clear') as mock_clear, \\
             patch.object(engine.console, 'print') as mock_print:
            
            engine._render_ui()
            
            mock_clear.assert_called_once()
            # Should print terminal size warning
            assert any("Terminal too small" in str(call) for call in mock_print.call_args_list)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_draft_text_generation(self):
        """Test draft generation with no content."""
        state = DraftingState()
        state.topic = None
        state.draft_content = []
        
        draft_text = state.get_draft_text()
        assert "My Thoughts" in draft_text  # Default topic
        assert "Items: 0" in draft_text
    
    def test_large_cluster_handling(self):
        """Test handling of clusters with many globules."""
        state = DraftingState()
        mock_cluster = Mock(id="large_cluster")
        large_globule_list = [Mock(text=f"Globule {i}") for i in range(1000)]
        
        state.clusters = [mock_cluster]
        state.globules_by_cluster = {"large_cluster": large_globule_list}
        state.selected_cluster_index = 0
        
        # Should handle large lists without issues
        globules = state.get_current_cluster_globules()
        assert len(globules) == 1000
    
    def test_unicode_text_handling(self):
        """Test handling of unicode text in globules."""
        state = DraftingState()
        unicode_text = "测试 unicode émojis 🚀 and symbols ∆∇∫"
        
        state.add_to_draft(unicode_text)
        draft_text = state.get_draft_text()
        assert unicode_text in draft_text


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])