# AI Note-Taking Assistant

A powerful AI-powered note-taking application that can process various types of content (documents, videos, text) and generate structured notes using advanced language models.

## Features

- Process YouTube videos and generate structured notes
- Process documents (PDF, DOCX, TXT) and extract key information
- Process raw text and generate organized notes
- Semantic search across all notes using vector embeddings
- Topic-based note organization and retrieval
- Automatic topic identification and categorization
- Support for private YouTube videos with authentication
- Progress tracking and logging
- Efficient resource management and cleanup

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
ai-note-taking-assistant/
├── src/
│   ├── document_processor.py    # Handles document processing
│   ├── video_processor.py      # Handles video processing
│   ├── text_processor.py       # Handles text processing
│   ├── note_manager.py         # Manages note storage and retrieval
│   └── note_assistant.py       # Main application class
├── tests/                      # Test files
├── data/                       # Data storage
│   └── notes/                  # Generated notes storage
├── logs/                       # Application logs
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-note-taking-assistant.git
cd ai-note-taking-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Anthropic API key
```

## Usage

### Command Line Interface

The application provides a command-line interface through `test_app.py`:

1. **Process a YouTube Video**
```bash
# Process a public YouTube video
python test_app.py --video "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a private video (requires authentication)
python test_app.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --auth-token "your_auth_token"

# Process with custom settings
python test_app.py --video "https://www.youtube.com/watch?v=VIDEO_ID" --playback-speed 1.5 --duration-limit "00:10:00"
```

2. **Process a Document**
```bash
# Process a PDF file
python test_app.py --document "path/to/your/document.pdf"

# Process a Word document
python test_app.py --document "path/to/your/document.docx"
```

3. **Process Text**
```bash
# Process text from a file
python test_app.py --text-file "path/to/your/text.txt"

# Process text directly
python test_app.py --text "Your text content here..."
```

### Python API

You can also use the application programmatically:

```python
from src.note_assistant import NoteAssistant

# Initialize the assistant
assistant = NoteAssistant()

# Process a YouTube video
result = assistant.process_video(
    video_url="https://www.youtube.com/watch?v=VIDEO_ID",
    auth_token=None,  # Optional: for private videos
    playback_speed=1.0,  # Optional: adjust playback speed
    duration_limit=None,  # Optional: limit video duration
    time_window=None,  # Optional: process specific time window
    progress_callback=lambda x, msg: print(f"Progress: {x}% - {msg}")
)

# Process a document
result = assistant.process_document(
    file_path="path/to/document.pdf",
    progress_callback=lambda x, msg: print(f"Progress: {x}% - {msg}")
)

# Process text
result = assistant.process_text(
    text="Your text content here...",
    progress_callback=lambda x, msg: print(f"Progress: {x}% - {msg}")
)

# Get notes by topic
notes = assistant.get_notes_by_topic("Artificial Intelligence")

# Merge notes from multiple topics
merged_notes = assistant.merge_notes([
    "Artificial Intelligence",
    "Machine Learning"
])
```

## Output Format

The notes are generated in a structured JSON format:

```json
{
    "main_topics": [
        "Topic 1",
        "Topic 2",
        "Topic 3"
    ],
    "notes": [
        {
            "topic": "Topic Name",
            "content": "Detailed notes about the topic",
            "key_points": ["Point 1", "Point 2"],
            "examples": ["Example 1", "Example 2"],
            "source": "video/document/text",
            "timestamp": "MM:SS"  // For videos, ISO format for documents/text
        }
    ]
}
```

## Features and Capabilities

1. **Video Processing**
   - Automatic transcript extraction
   - Support for private videos with authentication
   - Configurable playback speed
   - Duration limits and time windows
   - Progress tracking

2. **Document Processing**
   - Support for multiple formats (PDF, DOCX, TXT)
   - Automatic content extraction
   - Structured note generation
   - Progress tracking

3. **Text Processing**
   - Direct text input processing
   - File-based text processing
   - Structured note generation

4. **Note Management**
   - Vector-based semantic search
   - Topic-based organization
   - Note merging and summarization
   - Persistent storage

5. **Resource Management**
   - Automatic cleanup of resources
   - Context manager support
   - Proper error handling
   - Logging system

## Error Handling

The application includes comprehensive error handling:

```python
try:
    result = assistant.process_video("https://youtu.be/VIDEO_ID")
except FileNotFoundError:
    print("Video not found")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Logging

The application maintains detailed logs in the `logs` directory:
- Each run creates a new log file with timestamp
- Logs include processing progress, errors, and warnings
- Log files are named `app_YYYYMMDD_HHMMSS.log`

## Dependencies

Key dependencies include:
- `langchain-anthropic>=0.0.5` for LLM interactions
- `langchain-community>=0.0.13` for community components
- `langchain-core>=0.0.14` for core functionality
- `chromadb>=0.4.22` for vector storage
- `sentence-transformers>=2.2.2` for embeddings
- `youtube-transcript-api>=0.6.1` for video transcripts
- `selenium>=4.15.2` for web automation
- `python-dotenv>=1.0.0` for environment management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 