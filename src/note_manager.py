import os
import json
from typing import Dict, List, Optional
from datetime import datetime

class NoteManager:
    def __init__(self):
        """Initialize the note manager."""
        self.notes_dir = os.path.join("data", "notes")
        os.makedirs(self.notes_dir, exist_ok=True)

    def add_notes(self, notes: Dict, source: str) -> str:
        """Add notes to a new file and return the file path."""
        try:
            # Create a unique filename based on timestamp and source
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Clean the source string to create a valid filename
            clean_source = "".join(c for c in source if c.isalnum() or c in (' ', '-', '_')).strip()
            clean_source = clean_source.replace(' ', '_')[:50]  # Limit length
            filename = f"notes_{timestamp}_{clean_source}.json"
            filepath = os.path.join(self.notes_dir, filename)
            
            # Add metadata to the notes
            notes["metadata"] = {
                "source": source,
                "created_at": timestamp,
                "filename": filename
            }
            
            # Save notes to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(notes, f, indent=2, ensure_ascii=False)
            
            return filepath
            
        except Exception as e:
            raise Exception(f"Failed to save notes: {str(e)}")

    def get_notes_by_topic(self, topic: str, time_range: Optional[tuple] = None) -> List[Dict]:
        """Retrieve all notes for a specific topic, optionally filtered by time range."""
        try:
            all_notes = []
            
            # Read all note files
            for filename in os.listdir(self.notes_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.notes_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                    
                # Filter notes by topic
                for note in notes_data.get("notes", []):
                    if note.get("topic") == topic:
                        if time_range:
                            # Convert timestamp to datetime for comparison
                            note_time = datetime.fromisoformat(note["timestamp"].replace('Z', '+00:00'))
                            if time_range[0] <= note_time <= time_range[1]:
                                all_notes.append(note)
                        else:
                            all_notes.append(note)
            
            return all_notes
            
        except Exception as e:
            raise Exception(f"Failed to get notes for topic '{topic}': {str(e)}")

    def get_topic_timeline(self, topic: str) -> List[Dict]:
        """Get a chronological timeline of when topics appear in the video."""
        try:
            timeline = []
            
            # Read all note files
            for filename in os.listdir(self.notes_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.notes_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                    
                # Filter notes by topic and add to timeline
                for note in notes_data.get("notes", []):
                    if note.get("topic") == topic and "time_elapsed" in note:
                        timeline.append({
                            "time": note["time_elapsed"],
                            "content": note
                        })
            
            # Sort timeline by time
            timeline.sort(key=lambda x: x["time"])
            return timeline
            
        except Exception as e:
            raise Exception(f"Failed to get timeline for topic '{topic}': {str(e)}")

    def get_all_topics(self) -> List[str]:
        """Get a list of all topics."""
        try:
            topics = set()
            
            # Read all note files
            for filename in os.listdir(self.notes_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.notes_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                    
                # Add all topics to the set
                for note in notes_data.get("notes", []):
                    if "topic" in note:
                        topics.add(note["topic"])
            
            return sorted(list(topics))
            
        except Exception as e:
            raise Exception(f"Failed to get all topics: {str(e)}")

    def get_subtopics(self, topic: str) -> List[str]:
        """Get a list of all subtopics for a topic."""
        try:
            subtopics = set()
            
            # Read all note files
            for filename in os.listdir(self.notes_dir):
                if not filename.endswith('.json'):
                    continue
                    
                filepath = os.path.join(self.notes_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    notes_data = json.load(f)
                    
                # Add all subtopics for the given topic to the set
                for note in notes_data.get("notes", []):
                    if note.get("topic") == topic and "subtopics" in note:
                        subtopics.update(note["subtopics"])
            
            return sorted(list(subtopics))
            
        except Exception as e:
            raise Exception(f"Failed to get subtopics for topic '{topic}': {str(e)}") 