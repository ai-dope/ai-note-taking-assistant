"""
AI Note-Taking Assistant package.
"""

from .note_assistant import NoteAssistant
from .document_processor import DocumentProcessor
from .video_processor import VideoProcessor
from .note_manager import NoteManager

__all__ = ['NoteAssistant', 'DocumentProcessor', 'VideoProcessor', 'NoteManager'] 