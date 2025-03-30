import pytest
from unittest.mock import Mock, patch, MagicMock
from src.video_processor import VideoProcessor
from tests.test_config import SAMPLE_VIDEO_URL, TEST_USERNAME, TEST_PASSWORD
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import json

@pytest.fixture
def mock_llm():
    """Create a mock LLM with predefined responses."""
    mock = Mock()
    
    # Create a mock response class
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    # Create response objects with string content
    topics_response = MockResponse('{"main_topics": ["Topic 1", "Topic 2"]}')
    notes_response = MockResponse('{"notes": [{"topic": "Topic 1", "content": "Note content", "timestamp": "00:00"}]}')
    
    # Configure the mock to return different responses for different calls
    mock.invoke = Mock(side_effect=[topics_response] * 10)  # Allow multiple calls
    
    # Create a mock chain
    mock_chain = Mock()
    mock_chain.__or__ = Mock(return_value=mock)
    
    return mock

@pytest.fixture
def mock_embeddings():
    """Create mock embeddings."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock

@pytest.fixture
def video_processor(mock_llm):
    """Create a VideoProcessor instance with mocked dependencies."""
    processor = VideoProcessor(
        anthropic_api_key="test_anthropic_key",
        openai_api_key="test_openai_key",
        embeddings=Mock()
    )
    processor.llm = mock_llm
    return processor

@patch('selenium.webdriver.Chrome')
@patch('youtube_transcript_api.YouTubeTranscriptApi')
def test_process_video(mock_transcript_api, mock_chrome, video_processor):
    """Test basic video processing functionality."""
    # Mock transcript API
    mock_transcript_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
        {"text": "Sample transcript", "start": 0.0, "duration": 1.0}
    ]

    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver

    # Process video
    result = video_processor.process_video(SAMPLE_VIDEO_URL)

    # Verify the result structure
    assert "main_topics" in result
    assert "notes" in result
    assert "output_file" in result
    assert len(result["main_topics"]) > 0
    assert len(result["notes"]) > 0

@patch('selenium.webdriver.Chrome')
@patch('youtube_transcript_api.YouTubeTranscriptApi')
def test_process_video_with_auth(mock_transcript_api, mock_chrome, video_processor):
    """Test video processing with authentication."""
    # Mock transcript API
    mock_transcript_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
        {"text": "Sample transcript", "start": 0.0, "duration": 1.0}
    ]

    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver

    # Process video with auth token
    result = video_processor.process_video(SAMPLE_VIDEO_URL, auth_token="test_token")

    # Verify the result structure
    assert "main_topics" in result
    assert "notes" in result
    assert "output_file" in result

@patch('selenium.webdriver.Chrome')
@patch('youtube_transcript_api.YouTubeTranscriptApi')
def test_process_video_with_speed(mock_transcript_api, mock_chrome, video_processor):
    """Test processing a video with custom playback speed."""
    # Mock transcript API
    mock_transcript_api.list_transcripts.return_value.find_transcript.return_value.fetch.return_value = [
        {"text": "Sample transcript", "start": 0.0, "duration": 1.0}
    ]

    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver

    # Test video processing with custom speed
    result = video_processor.process_video(SAMPLE_VIDEO_URL, playback_speed=2.0)

    # Verify the result structure
    assert "main_topics" in result
    assert "notes" in result
    assert "output_file" in result

@patch('selenium.webdriver.Chrome')
@patch('youtube_transcript_api.YouTubeTranscriptApi')
def test_process_video_no_transcript(mock_transcript_api, mock_chrome, video_processor):
    """Test handling of video without transcript."""
    # Mock transcript API to raise an exception
    mock_transcript_api.list_transcripts.side_effect = Exception("No transcript available")

    # Mock Selenium elements
    mock_driver = Mock()
    mock_chrome.return_value = mock_driver

    # Test video processing without transcript
    with pytest.raises(Exception) as exc_info:
        video_processor.process_video(SAMPLE_VIDEO_URL)

    assert "No transcript available" in str(exc_info.value)

def test_clean_json_string(video_processor):
    """Test the clean_json_string method."""
    # Test with valid JSON
    valid_json = '{"key": "value"}'
    assert video_processor.clean_json_string(valid_json) == valid_json

    # Test with JSON containing unquoted properties
    unquoted_json = '{key: "value"}'
    expected = '{"key": "value"}'
    assert video_processor.clean_json_string(unquoted_json) == expected

    # Test with JSON in code block
    code_block_json = '```json\n{"key": "value"}\n```'
    assert video_processor.clean_json_string(code_block_json) == valid_json 