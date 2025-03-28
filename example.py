from src.note_assistant import NoteAssistant

def main():
    # Initialize the note-taking assistant
    assistant = NoteAssistant()
    
    # Example 1: Process a document
    try:
        document_path = "path/to/your/document.docx"  # Replace with actual document path
        notes_file = assistant.process_document(document_path)
        print(f"Document processed successfully. Notes saved to: {notes_file}")
    except Exception as e:
        print(f"Error processing document: {e}")
    
    # Example 2: Process a video
    try:
        video_url = "https://example.com/video"  # Replace with actual video URL
        notes_file = assistant.process_video(
            url=video_url,
            username="user",  # Optional
            password="pass",  # Optional
            playback_speed=1.5  # Optional
        )
        print(f"Video processed successfully. Notes saved to: {notes_file}")
    except Exception as e:
        print(f"Error processing video: {e}")
    
    # Example 3: Get all topics
    topics = assistant.get_all_topics()
    print("\nAll topics:")
    for topic in topics:
        print(f"- {topic}")
        # Get subtopics for each topic
        subtopics = assistant.get_subtopics(topic)
        print("  Subtopics:")
        for subtopic in subtopics:
            print(f"  - {subtopic}")
    
    # Example 4: Get notes for a specific topic
    if topics:
        topic = topics[0]  # Get notes for the first topic
        notes = assistant.get_notes_by_topic(topic)
        print(f"\nNotes for topic '{topic}':")
        for note in notes:
            print(f"- Source: {note['source']}")
            print(f"  Timestamp: {note['timestamp']}")

if __name__ == "__main__":
    main() 