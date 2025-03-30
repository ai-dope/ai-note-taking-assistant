import os
import logging
import argparse
from dotenv import load_dotenv
from src.note_assistant import NoteAssistant
from tqdm import tqdm
import sys
from datetime import datetime

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate unique log filename with timestamp
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging to write to file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stderr)  # Only show errors in console
        ]
    )
    
    # Force reconfiguration of all loggers
    logging.getLogger().handlers = []
    logging.getLogger().addHandler(logging.FileHandler(log_filename))
    
    # Configure all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.addHandler(logging.FileHandler(log_filename))
        logger.propagate = False
    
    return log_filename

def update_progress(x: int, stage: str) -> None:
    """Update the progress bar with the current stage."""
    pbar.set_description(f"Overall Progress: {stage}")
    pbar.n = x
    pbar.refresh()

def main():
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Process various types of content and generate notes.')
    parser.add_argument('--video', help='URL of the video to process')
    parser.add_argument('--document', help='Path to the document to process')
    parser.add_argument('--text', help='Text to process')
    parser.add_argument('--text-file', help='Path to the text file to process')
    parser.add_argument('--playback-speed', type=float, help='Playback speed for video processing')
    parser.add_argument('--auth-token', help='Authentication token for video processing')
    parser.add_argument('--duration-limit', help='Duration limit in HH:MM:SS format (e.g., "00:05:00" for 5 minutes)')
    parser.add_argument('--time-window', nargs=2, help='Time window in HH:MM:SS format (e.g., "00:06:00" "00:08:24" for 6:00-8:24)')
    
    args = parser.parse_args()
    
    # Set up logging
    log_filename = setup_logging()
    logging.info("Initializing NoteAssistant")
    
    # Initialize the assistant
    assistant = NoteAssistant()
    
    # Create progress bar
    global pbar
    pbar = tqdm(total=100, desc="Overall Progress", file=sys.stdout)
    
    try:
        if args.video:
            logging.info(f"Processing video: {args.video}")
            result = assistant.process_video(
                args.video,
                auth_token=args.auth_token,
                playback_speed=args.playback_speed,
                duration_limit=args.duration_limit,
                time_window=tuple(args.time_window) if args.time_window else None,
                progress_callback=update_progress
            )
            print("\nSuccessfully processed video.")
            print(f"Notes saved to: {result['output_file']}")
            print("\nAvailable topics:")
            for topic in result["main_topics"]:
                print(f"- {topic}")
            
            print(f"\nLog file: {log_filename}")
            
        elif args.document:
            logging.info(f"Processing document: {args.document}")
            result = assistant.process_document(
                args.document,
                progress_callback=update_progress
            )
            print(f"\nSuccessfully processed document. Notes saved to: {result['output_file']}")
            
        elif args.text:
            logging.info("Processing text input")
            result = assistant.process_text(
                args.text,
                progress_callback=update_progress
            )
            print(f"\nSuccessfully processed text. Notes saved to: {result['output_file']}")
            
        elif args.text_file:
            logging.info(f"Processing text file: {args.text_file}")
            result = assistant.process_text_file(
                args.text_file,
                progress_callback=update_progress
            )
            print(f"\nSuccessfully processed text file. Notes saved to: {result['output_file']}")
            
        else:
            parser.print_help()
            return
        
        # Print available topics
        if result and 'main_topics' in result:
            print("\nAvailable topics:")
            for topic in result['main_topics']:
                print(f"- {topic}")
        
        print(f"\nLog file: {log_filename}")
        
    except Exception as e:
        logging.error(f"Error processing content: {str(e)}")
        print(f"\nError: {str(e)}")
        sys.exit(1)
    finally:
        pbar.close()

if __name__ == '__main__':
    main() 