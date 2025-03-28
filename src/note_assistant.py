import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from .document_processor import DocumentProcessor
from .video_processor import VideoProcessor
from .note_manager import NoteManager

class NoteAssistant:
    def __init__(self):
        load_dotenv()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
        self.document_processor = DocumentProcessor(self.anthropic_api_key)
        self.video_processor = VideoProcessor(self.anthropic_api_key)
        self.note_manager = NoteManager()

    def process_document(self, file_path: str) -> str:
        """Process a document and store the notes."""
        notes = self.document_processor.process_document(file_path)
        return self.note_manager.add_notes(notes, f"Document: {file_path}")

    def process_video(self, video_url: str, auth_token: Optional[str] = None, playback_speed: float = 2.0) -> str:
        """Process a video and store the notes.
        
        Args:
            video_url: URL of the YouTube video
            auth_token: Optional authentication token for private videos
            playback_speed: Desired playback speed (default: 2.0x)
        """
        notes = self.video_processor.process_video(video_url, auth_token, playback_speed)
        return self.note_manager.add_notes(notes, f"Video: {video_url}")

    def get_notes_by_topic(self, topic: str, time_range: Optional[tuple] = None) -> list:
        """Retrieve all notes for a specific topic, optionally filtered by time range."""
        return self.note_manager.get_notes_by_topic(topic, time_range)

    def get_topic_timeline(self, topic: str) -> list:
        """Get a chronological timeline of when topics appear in the video."""
        return self.note_manager.get_topic_timeline(topic)

    def get_all_topics(self) -> list:
        """Get a list of all topics."""
        return self.note_manager.get_all_topics()

    def get_subtopics(self, topic: str) -> list:
        """Get a list of all subtopics for a topic."""
        return self.note_manager.get_subtopics(topic)

    def merge_notes(self, existing_file: str, new_notes: Dict) -> str:
        """Merge new notes with existing notes."""
        return self.note_manager.merge_notes(existing_file, new_notes) 