import pytest
import json
import os
from pathlib import Path
from src.note_manager import NoteManager
from .test_config import SAMPLE_NOTES, TEST_DATA_DIR

@pytest.fixture
def note_manager():
    """Create a NoteManager instance with a test directory."""
    test_dir = TEST_DATA_DIR / "notes"
    test_dir.mkdir(exist_ok=True)
    return NoteManager(base_dir=str(test_dir))

@pytest.fixture(autouse=True)
def cleanup_test_files(note_manager):
    """Clean up test files after each test."""
    yield
    # Remove all files in the test directory
    for file in note_manager.base_dir.glob("*"):
        file.unlink()
    if note_manager.index_file.exists():
        note_manager.index_file.unlink()

def test_add_notes(note_manager):
    """Test adding new notes."""
    note_file = note_manager.add_notes(SAMPLE_NOTES, "Test Source")
    
    assert os.path.exists(note_file)
    with open(note_file, 'r') as f:
        saved_data = json.load(f)
        assert saved_data["source"] == "Test Source"
        assert saved_data["content"] == SAMPLE_NOTES

def test_get_notes_by_topic(note_manager):
    """Test retrieving notes by topic."""
    # Add test notes
    note_manager.add_notes(SAMPLE_NOTES, "Test Source")
    
    # Get notes for the test topic
    notes = note_manager.get_notes_by_topic("Test Topic")
    assert len(notes) == 1
    assert notes[0]["source"] == "Test Source"

def test_get_all_topics(note_manager):
    """Test getting all topics."""
    # Add test notes
    note_manager.add_notes(SAMPLE_NOTES, "Test Source")
    
    # Get all topics
    topics = note_manager.get_all_topics()
    assert "Test Topic" in topics

def test_get_subtopics(note_manager):
    """Test getting subtopics for a topic."""
    # Add test notes
    note_manager.add_notes(SAMPLE_NOTES, "Test Source")
    
    # Get subtopics
    subtopics = note_manager.get_subtopics("Test Topic")
    assert "Test Subtopic" in subtopics

def test_merge_notes(note_manager):
    """Test merging notes."""
    # Add initial notes
    initial_file = note_manager.add_notes(SAMPLE_NOTES, "Initial Source")
    
    # Create new notes to merge
    new_notes = {
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 3"]
                    }
                ]
            }
        ]
    }
    
    # Merge notes
    merged_file = note_manager.merge_notes(initial_file, new_notes)
    
    # Verify merged content
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)
        points = merged_data["content"]["main_topics"][0]["subtopics"][0]["points"]
        assert len(points) == 3
        assert "Test Point 3" in points

def test_merge_notes_new_topic(note_manager):
    """Test merging notes with a new topic."""
    # Add initial notes
    initial_file = note_manager.add_notes(SAMPLE_NOTES, "Initial Source")
    
    # Create new notes with a different topic
    new_notes = {
        "main_topics": [
            {
                "topic": "New Topic",
                "subtopics": [
                    {
                        "subtopic": "New Subtopic",
                        "points": ["New Point 1"]
                    }
                ]
            }
        ]
    }
    
    # Merge notes
    merged_file = note_manager.merge_notes(initial_file, new_notes)
    
    # Verify merged content
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)
        topics = [t["topic"] for t in merged_data["content"]["main_topics"]]
        assert "Test Topic" in topics
        assert "New Topic" in topics

def test_index_persistence(note_manager):
    """Test that the index is properly persisted."""
    # Add test notes
    note_manager.add_notes(SAMPLE_NOTES, "Test Source")
    
    # Create a new NoteManager instance
    new_manager = NoteManager(base_dir=str(note_manager.base_dir))
    
    # Verify that the topics are preserved
    assert "Test Topic" in new_manager.get_all_topics() 