import os
import logging
import argparse
from dotenv import load_dotenv
from src.note_assistant import NoteAssistant

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process content for note-taking')
    parser.add_argument('--video', help='URL of the YouTube video to process')
    parser.add_argument('--document', help='Path to the document to process')
    parser.add_argument('--text', help='Text content to process')
    parser.add_argument('--text-file', help='Path to text file to process')
    parser.add_argument('--auth-token', help='Authentication token for private videos')
    parser.add_argument('--playback-speed', type=float, default=2.0,
                      help='Video playback speed (default: 2.0x)')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Initialize the note assistant
    print("Initializing NoteAssistant...")
    assistant = NoteAssistant()
    
    try:
        if args.video:
            print(f"\nProcessing video: {args.video}")
            print(f"Playback speed: {args.playback_speed}x")
            logger.debug("Starting video processing...")
            assistant.process_video(args.video, args.auth_token, args.playback_speed)
            logger.debug("Video processing completed successfully")
            print("Video processed successfully!")
            
        elif args.document:
            print(f"\nProcessing document: {args.document}")
            logger.debug("Starting document processing...")
            assistant.process_document(args.document)
            logger.debug("Document processing completed successfully")
            print("Document processed successfully!")
            
        elif args.text:
            print("\nProcessing text content...")
            logger.debug("Starting text processing...")
            assistant.process_text(args.text)
            logger.debug("Text processing completed successfully")
            print("Text processed successfully!")
            
        elif args.text_file:
            print(f"\nProcessing text file: {args.text_file}")
            logger.debug("Starting text file processing...")
            with open(args.text_file, 'r') as f:
                text_content = f.read()
            assistant.process_text(text_content)
            logger.debug("Text file processing completed successfully")
            print("Text file processed successfully!")
            
        else:
            print("No input specified. Please provide --video, --document, --text, or --text-file argument.")
            return
        
        # Get available topics
        print("\nAvailable topics:")
        topics = assistant.get_all_topics()
        for topic in topics:
            print(f"- {topic}")
        
        # Get notes for the first topic
        if topics:
            first_topic = topics[0]
            print(f"\nGetting notes for '{first_topic}' topic:")
            notes = assistant.get_notes_by_topic(first_topic)
            for note in notes:
                print(f"\nContent: {note['content']}")
                print(f"Source: {note['source']}")
                print(f"Timestamp: {note['timestamp']}")
            
            # Show timeline for the topic
            print(f"\nTimeline for '{first_topic}':")
            timeline = assistant.get_topic_timeline(first_topic)
            for entry in timeline:
                minutes = int(entry['time'] // 60)
                seconds = int(entry['time'] % 60)
                print(f"\nTime: {minutes:02d}:{seconds:02d}")
                print(f"Content: {entry['content']['content']}")
            
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 