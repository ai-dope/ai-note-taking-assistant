import os
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from .vector_store import VectorStore

class NoteManager:
    def __init__(self, base_dir: str = "./data/notes"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load the note index from disk."""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        # Create empty index if file doesn't exist
        if not os.path.exists(self.index_file):
            self.index = {
                "topics": {},
                "notes": [],
                "last_updated": datetime.now().isoformat()
            }
            self._save_index()
            return

        with open(self.index_file, 'r') as f:
            self.index = json.load(f)

    def _save_index(self):
        """Save the index file."""
        self.index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)

    def add_notes(self, notes: Dict, source: str) -> str:
        """Add new notes to the system."""
        # Create a new note file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_file = self.base_dir / f"notes_{timestamp}.json"
        
        note_data = {
            "source": source,
            "timestamp": timestamp,
            "content": notes
        }
        
        with open(note_file, 'w') as f:
            json.dump(note_data, f, indent=2)
        
        # Update index
        self._update_index(notes, str(note_file))
        return str(note_file)

    def _update_index(self, notes: Dict, note_file: str):
        """Update the index with new notes."""
        # Handle main_topics as a list of strings
        for topic in notes.get("main_topics", []):
            if topic not in self.index["topics"]:
                self.index["topics"][topic] = {
                    "files": [],
                    "subtopics": [],
                    "time_ranges": []  # Store time ranges for each topic
                }
            
            # Add note file to topic
            if note_file not in self.index["topics"][topic]["files"]:
                self.index["topics"][topic]["files"].append(note_file)
        
        # Handle notes as a list of dictionaries
        for note in notes.get("notes", []):
            topic = note.get("topic")
            if topic and topic not in self.index["topics"]:
                self.index["topics"][topic] = {
                    "files": [],
                    "subtopics": [],
                    "time_ranges": []
                }
            
            if topic:
                if note_file not in self.index["topics"][topic]["files"]:
                    self.index["topics"][topic]["files"].append(note_file)
                
                # Store time range for this topic
                time_elapsed = note.get("time_elapsed", 0)
                self.index["topics"][topic]["time_ranges"].append({
                    "file": note_file,
                    "time": time_elapsed
                })
        
        self._save_index()

    def get_notes_by_topic(self, topic: str, time_range: Optional[tuple] = None) -> List[Dict]:
        """Retrieve all notes for a specific topic, optionally filtered by time range."""
        if topic not in self.index["topics"]:
            return []
        
        notes = []
        for note_file in self.index["topics"][topic]["files"]:
            with open(note_file, 'r') as f:
                note_data = json.load(f)
                # Filter by time range if specified
                if time_range:
                    start_time, end_time = time_range
                    note_time = note_data.get("content", {}).get("time_elapsed", 0)
                    if start_time <= note_time <= end_time:
                        notes.append(note_data)
                else:
                    notes.append(note_data)
        
        return notes

    def get_all_topics(self) -> List[str]:
        """Get a list of all topics."""
        return list(self.index["topics"].keys())

    def get_subtopics(self, topic: str) -> List[str]:
        """Get a list of all subtopics for a topic."""
        if topic not in self.index["topics"]:
            return []
        return list(self.index["topics"][topic]["subtopics"])

    def merge_notes(self, existing_file: str, new_notes: Dict) -> str:
        """Merge new notes with existing notes."""
        with open(existing_file, 'r') as f:
            existing_data = json.load(f)
        
        # Merge the content
        merged_content = self._merge_content(existing_data["content"], new_notes)
        
        # Create new merged file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_file = self.base_dir / f"merged_{timestamp}.json"
        
        merged_data = {
            "source": f"Merged from {existing_data['source']}",
            "timestamp": timestamp,
            "content": merged_content
        }
        
        with open(merged_file, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        # Update index
        self._update_index(merged_content, str(merged_file))
        return str(merged_file)

    def _merge_content(self, existing: Dict, new: Dict) -> Dict:
        """Merge existing and new content, combining topics and subtopics."""
        merged = {"main_topics": []}
        existing_topics = {t["topic"]: t for t in existing.get("main_topics", [])}
        new_topics = {t["topic"]: t for t in new.get("main_topics", [])}
        
        # Process all topics
        all_topics = set(existing_topics.keys()) | set(new_topics.keys())
        for topic in all_topics:
            if topic in existing_topics and topic in new_topics:
                # Merge existing and new topic
                merged_topic = self._merge_topic(existing_topics[topic], new_topics[topic])
            elif topic in existing_topics:
                merged_topic = existing_topics[topic]
            else:
                merged_topic = new_topics[topic]
            
            merged["main_topics"].append(merged_topic)
        
        return merged

    def _merge_topic(self, existing: Dict, new: Dict) -> Dict:
        """Merge two topics, combining their subtopics and points."""
        merged = {
            "topic": existing["topic"],
            "subtopics": []
        }
        
        existing_subtopics = {s["subtopic"]: s for s in existing.get("subtopics", [])}
        new_subtopics = {s["subtopic"]: s for s in new.get("subtopics", [])}
        
        # Process all subtopics
        all_subtopics = set(existing_subtopics.keys()) | set(new_subtopics.keys())
        for subtopic in all_subtopics:
            if subtopic in existing_subtopics and subtopic in new_subtopics:
                # Merge existing and new subtopic
                merged_subtopic = {
                    "subtopic": subtopic,
                    "points": list(set(
                        existing_subtopics[subtopic]["points"] +
                        new_subtopics[subtopic]["points"]
                    ))
                }
            elif subtopic in existing_subtopics:
                merged_subtopic = existing_subtopics[subtopic]
            else:
                merged_subtopic = new_subtopics[subtopic]
            
            merged["subtopics"].append(merged_subtopic)
        
        return merged 

    def get_topic_timeline(self, topic: str) -> List[Dict]:
        """Get a chronological timeline of when topics appear in the video."""
        if topic not in self.index["topics"]:
            return []
        
        timeline = []
        for time_range in self.index["topics"][topic]["time_ranges"]:
            with open(time_range["file"], 'r') as f:
                note_data = json.load(f)
                timeline.append({
                    "time": time_range["time"],
                    "content": note_data
                })
        
        # Sort by time
        timeline.sort(key=lambda x: x["time"])
        return timeline 