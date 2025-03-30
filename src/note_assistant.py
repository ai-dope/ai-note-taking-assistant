import os
import json
from typing import Dict, List, Optional, Callable
from dotenv import load_dotenv
from .document_processor import DocumentProcessor
from .video_processor import VideoProcessor
from .note_manager import NoteManager
from .text_processor import TextProcessor
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
from langchain_huggingface import HuggingFaceEmbeddings

class NoteAssistant:
    def __init__(self):
        """Initialize the note assistant with API keys."""
        # Load API keys from environment variables
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
            
        # Initialize video processor with both API keys
        self.video_processor = VideoProcessor(
            anthropic_api_key=self.anthropic_api_key,
            openai_api_key=self.openai_api_key or "",  # Use empty string if None
            embeddings=self.embeddings
        )
        
        self.document_processor = DocumentProcessor(self.anthropic_api_key)
        self.text_processor = TextProcessor(self.anthropic_api_key)
        self.note_manager = NoteManager()
        
        # Initialize LLM for note merging
        self.llm = ChatAnthropic(
            api_key=SecretStr(self.anthropic_api_key),
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            timeout=30,
            stop=None
        )
        
        # Create notes directory if it doesn't exist
        os.makedirs("data/notes", exist_ok=True)

    def process_document(self, file_path: str,
                        progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process a document and generate structured notes."""
        result = self.document_processor.process_document(
            file_path,
            progress_callback=progress_callback
        )
        
        # Save notes to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/notes/document_notes_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        result['output_file'] = output_file
        return result

    def process_video(self, video_url: str, auth_token: Optional[str] = None,
                     playback_speed: Optional[float] = None,
                     duration_limit: Optional[str] = None,
                     time_window: Optional[tuple[str, str]] = None,
                     progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process a video and generate structured notes."""
        try:
            # Create notes directory if it doesn't exist
            os.makedirs("data/notes", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/notes/video_notes_{timestamp}.json"
            
            result = self.video_processor.process_video(
                video_url,
                auth_token=auth_token,
                playback_speed=playback_speed,
                duration_limit=duration_limit,
                time_window=time_window,
                progress_callback=progress_callback
            )
            
            # Save notes to file with complete data
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "video_url": video_url,
                    "main_topics": result["main_topics"],
                    "notes": result["notes"],
                    "processed_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            result["output_file"] = output_file
            return result
            
        except Exception as e:
            raise Exception(f"Failed to process video: {str(e)}")

    def process_text(self, text: str,
                    progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process text and generate structured notes."""
        result = self.text_processor.process_text(
            text,
            progress_callback=progress_callback
        )
        
        # Save notes to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/notes/text_notes_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        result['output_file'] = output_file
        return result

    def process_text_file(self, file_path: str,
                         progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process a text file and generate structured notes."""
        with open(file_path, 'r') as f:
            text = f.read()
        
        return self.process_text(text, progress_callback)

    def get_notes_by_topic(self, topic: str) -> List[Dict]:
        """Retrieve notes for a specific topic from all processors."""
        notes = []
        
        # Get notes from each processor
        video_notes = self.video_processor.get_notes_by_topic(topic)
        document_notes = self.document_processor.get_notes_by_topic(topic)
        text_notes = self.text_processor.get_notes_by_topic(topic)
        
        # Combine all notes
        notes.extend(video_notes)
        notes.extend(document_notes)
        notes.extend(text_notes)
        
        return notes

    def get_topic_timeline(self, topic: str) -> list:
        """Get a chronological timeline of when topics appear in the video."""
        try:
            return self.note_manager.get_topic_timeline(topic)
        except Exception as e:
            raise Exception(f"Failed to get timeline for topic '{topic}': {str(e)}")

    def get_all_topics(self) -> list:
        """Get a list of all topics."""
        try:
            return self.note_manager.get_all_topics()
        except Exception as e:
            raise Exception(f"Failed to get all topics: {str(e)}")

    def get_subtopics(self, topic: str) -> list:
        """Get a list of all subtopics for a topic."""
        try:
            return self.note_manager.get_subtopics(topic)
        except Exception as e:
            raise Exception(f"Failed to get subtopics for topic '{topic}': {str(e)}")

    def merge_notes(self, topics: List[str]) -> Dict:
        """Merge notes from multiple topics into a cohesive summary."""
        # Get notes for all topics
        all_notes = []
        for topic in topics:
            topic_notes = self.get_notes_by_topic(topic)
            all_notes.extend(topic_notes)
        
        if not all_notes:
            return {"summary": "No notes found for the specified topics."}
        
        # Create a summary using the LLM
        prompt = PromptTemplate(
            input_variables=["notes"],
            template="""
            You are an expert at synthesizing information. Given the following notes from multiple topics,
            please create a cohesive summary that connects the key ideas and highlights important relationships.
            
            Notes: {notes}
            
            Summary:
            """
        )
        
        chain = prompt | self.llm
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        } 