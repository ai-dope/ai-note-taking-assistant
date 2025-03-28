import pytest
import os
from unittest.mock import Mock, patch
from src.document_processor import DocumentProcessor
from .test_config import SAMPLE_DOCX, SAMPLE_PDF, create_test_documents

@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance with a mock OpenAI API key."""
    return DocumentProcessor(openai_api_key="test_key")

@pytest.fixture(autouse=True)
def setup_test_documents():
    """Create test documents before each test."""
    create_test_documents()

def test_process_docx(document_processor):
    """Test processing a DOCX file."""
    notes = document_processor.process_document(str(SAMPLE_DOCX))
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    assert len(notes["main_topics"]) > 0

def test_process_pdf(document_processor):
    """Test processing a PDF file."""
    notes = document_processor.process_document(str(SAMPLE_PDF))
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    assert len(notes["main_topics"]) > 0

def test_invalid_file_type(document_processor):
    """Test handling of invalid file type."""
    with pytest.raises(ValueError):
        document_processor.process_document("test.txt")

def test_file_not_found(document_processor):
    """Test handling of non-existent file."""
    with pytest.raises(FileNotFoundError):
        document_processor.process_document("nonexistent.docx")

@patch('langchain.chat_models.ChatOpenAI')
def test_generate_notes(mock_chat_openai, document_processor):
    """Test note generation with mock LLM response."""
    # Mock the LLM response
    mock_response = {
        "main_topics": [
            {
                "topic": "Test Topic",
                "subtopics": [
                    {
                        "subtopic": "Test Subtopic",
                        "points": ["Test Point 1", "Test Point 2"]
                    }
                ]
            }
        ]
    }
    mock_chain = Mock()
    mock_chain.run.return_value = str(mock_response)
    mock_chat_openai.return_value = mock_chain

    # Test note generation
    chunks = ["Test chunk 1", "Test chunk 2"]
    notes = document_processor._generate_notes(chunks)
    
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    assert len(notes["main_topics"]) == 1
    assert notes["main_topics"][0]["topic"] == "Test Topic"

def test_merge_notes(document_processor):
    """Test merging of notes."""
    existing_notes = {
        "main_topics": [
            {
                "topic": "Topic 1",
                "subtopics": [
                    {
                        "subtopic": "Subtopic 1",
                        "points": ["Point 1", "Point 2"]
                    }
                ]
            }
        ]
    }
    
    new_notes = {
        "main_topics": [
            {
                "topic": "Topic 1",
                "subtopics": [
                    {
                        "subtopic": "Subtopic 1",
                        "points": ["Point 2", "Point 3"]
                    }
                ]
            },
            {
                "topic": "Topic 2",
                "subtopics": [
                    {
                        "subtopic": "Subtopic 1",
                        "points": ["Point 1"]
                    }
                ]
            }
        ]
    }
    
    merged = document_processor.merge_notes(existing_notes, new_notes)
    assert len(merged["main_topics"]) == 2
    assert len(merged["main_topics"][0]["subtopics"][0]["points"]) == 3
    assert merged["main_topics"][1]["topic"] == "Topic 2" 