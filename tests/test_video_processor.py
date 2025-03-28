import pytest
from unittest.mock import Mock, patch
from src.video_processor import VideoProcessor
from .test_config import SAMPLE_VIDEO_URL, TEST_USERNAME, TEST_PASSWORD

@pytest.fixture
def video_processor():
    """Create a VideoProcessor instance with a mock OpenAI API key."""
    return VideoProcessor(openai_api_key="test_key")

@patch('selenium.webdriver.Chrome')
def test_process_video(mock_chrome, video_processor):
    """Test processing a video with mock Selenium."""
    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    
    # Set up mock driver behavior
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver
    
    # Test video processing
    notes = video_processor.process_video(SAMPLE_VIDEO_URL)
    
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    assert len(notes["main_topics"]) > 0
    mock_driver.quit.assert_called_once()

@patch('selenium.webdriver.Chrome')
def test_process_video_with_auth(mock_chrome, video_processor):
    """Test processing a video with authentication."""
    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    
    # Set up mock driver behavior
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver
    
    # Test video processing with authentication
    notes = video_processor.process_video(
        SAMPLE_VIDEO_URL,
        username=TEST_USERNAME,
        password=TEST_PASSWORD
    )
    
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    mock_driver.quit.assert_called_once()

@patch('selenium.webdriver.Chrome')
def test_process_video_with_speed(mock_chrome, video_processor):
    """Test processing a video with custom playback speed."""
    # Mock Selenium elements
    mock_driver = Mock()
    mock_element = Mock()
    mock_element.text = "Sample transcript"
    
    # Set up mock driver behavior
    mock_driver.find_element.return_value = mock_element
    mock_chrome.return_value = mock_driver
    
    # Test video processing with custom speed
    notes = video_processor.process_video(
        SAMPLE_VIDEO_URL,
        playback_speed=2.0
    )
    
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    mock_driver.quit.assert_called_once()

@patch('selenium.webdriver.Chrome')
def test_process_video_no_transcript(mock_chrome, video_processor):
    """Test handling of video without transcript."""
    # Mock Selenium elements
    mock_driver = Mock()
    mock_chrome.return_value = mock_driver
    
    # Test video processing without transcript
    with pytest.raises(Exception) as exc_info:
        video_processor.process_video(SAMPLE_VIDEO_URL)
    
    assert "No transcript available" in str(exc_info.value)
    mock_driver.quit.assert_called_once()

@patch('langchain.chat_models.ChatOpenAI')
def test_generate_notes(mock_chat_openai, video_processor):
    """Test note generation from transcript."""
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
    transcript = "Sample transcript text"
    notes = video_processor._generate_notes(transcript)
    
    assert isinstance(notes, dict)
    assert "main_topics" in notes
    assert len(notes["main_topics"]) == 1
    assert notes["main_topics"][0]["topic"] == "Test Topic" 