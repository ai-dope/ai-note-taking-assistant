import pytest
import os
from unittest.mock import patch, Mock
from src.note_assistant import NoteAssistant
from .test_config import (
    SAMPLE_DOCX, SAMPLE_PDF, SAMPLE_VIDEO_URL,
    TEST_USERNAME, TEST_PASSWORD, create_test_documents
)

@pytest.fixture
def note_assistant():
    """Create a NoteAssistant instance with mock API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        return NoteAssistant()

@pytest.fixture(autouse=True)
def setup_test_documents():
    """Create test documents before each test."""
    create_test_documents()

@patch('src.document_processor.DocumentProcessor')
@patch('src.video_processor.VideoProcessor')
def test_process_document(mock_video_processor, mock_document_processor, note_assistant):
    """Test processing a document through the main interface."""
    # Mock document processor
    mock_doc_processor = Mock()
    mock_doc_processor.process_document.return_value = {
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 1"]
                    }
                ]
            }
        ]
    }
    mock_document_processor.return_value = mock_doc_processor
    
    # Process document
    notes_file = note_assistant.process_document(str(SAMPLE_DOCX))
    
    assert os.path.exists(notes_file)
    mock_doc_processor.process_document.assert_called_once_with(str(SAMPLE_DOCX))

@patch('src.document_processor.DocumentProcessor')
@patch('src.video_processor.VideoProcessor')
def test_process_video(mock_video_processor, mock_document_processor, note_assistant):
    """Test processing a video through the main interface."""
    # Mock video processor
    mock_vid_processor = Mock()
    mock_vid_processor.process_video.return_value = {
        "main_topics": [
            {
                "topic": "Video Topic",
                "subtopics": [
                    {
                        "subtopic": "Video Subtopic",
                        "points": ["Video Point 1"]
                    }
                ]
            }
        ]
    }
    mock_video_processor.return_value = mock_vid_processor
    
    # Process video
    notes_file = note_assistant.process_video(
        SAMPLE_VIDEO_URL,
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
        playback_speed=1.5
    )
    
    assert os.path.exists(notes_file)
    mock_vid_processor.process_video.assert_called_once_with(
        SAMPLE_VIDEO_URL,
        username=TEST_USERNAME,
        password=TEST_PASSWORD,
        playback_speed=1.5
    )

@patch('src.document_processor.DocumentProcessor')
@patch('src.video_processor.VideoProcessor')
def test_get_notes_by_topic(mock_video_processor, mock_document_processor, note_assistant):
    """Test retrieving notes by topic through the main interface."""
    # Add some test notes
    note_assistant.note_manager.add_notes({
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 1"]
                    }
                ]
            }
        ]
    }, "Test Source")
    
    # Get notes by topic
    notes = note_assistant.get_notes_by_topic("Test Topic")
    assert len(notes) == 1
    assert notes[0]["source"] == "Test Source"

@patch('src.document_processor.DocumentProcessor')
@patch('src.video_processor.VideoProcessor')
def test_merge_notes(mock_video_processor, mock_document_processor, note_assistant):
    """Test merging notes through the main interface."""
    # Add initial notes
    initial_file = note_assistant.note_manager.add_notes({
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 1"]
                    }
                ]
            }
        ]
    }, "Initial Source")
    
    # Create new notes to merge
    new_notes = {
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 2"]
                    }
                ]
            }
        ]
    }
    
    # Merge notes
    merged_file = note_assistant.merge_notes(initial_file, new_notes)
    
    # Verify merged content
    with open(merged_file, 'r') as f:
        merged_data = json.load(f)
        points = merged_data["content"]["main_topics"][0]["subtopics"][0]["points"]
        assert len(points) == 2
        assert "Test Point 1" in points
        assert "Test Point 2" in points

@patch('src.document_processor.DocumentProcessor')
@patch('src.video_processor.VideoProcessor')
def test_error_handling(mock_video_processor, mock_document_processor, note_assistant):
    """Test error handling in the main interface."""
    # Mock document processor to raise an exception
    mock_doc_processor = Mock()
    mock_doc_processor.process_document.side_effect = Exception("Test error")
    mock_document_processor.return_value = mock_doc_processor
    
    # Test error handling
    with pytest.raises(Exception) as exc_info:
        note_assistant.process_document(str(SAMPLE_DOCX))
    
    assert "Test error" in str(exc_info.value) 