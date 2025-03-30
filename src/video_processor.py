import os
from typing import Dict, List, Optional, Callable, Union, Any, TypeVar, cast, Type
from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from requests.exceptions import RequestException
from requests.models import Response
import time
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from youtube_transcript_api._api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from datetime import datetime
import re
from openai import OpenAI
from pydantic import SecretStr
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableSequence

T = TypeVar('T')

def safe_json_loads(s: str) -> Dict:
    """Safely load JSON string with better error handling."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        s = s.strip()
        # Replace single quotes with double quotes
        s = s.replace("'", '"')
        # Ensure property names are properly quoted
        s = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            raise

class APIError(Exception):
    """Custom exception class for API errors with response handling."""
    def __init__(self, message: str, response: Optional[Response] = None):
        super().__init__(message)
        self.response = response
        self.status_code: Optional[int] = response.status_code if response is not None else None

class VideoProcessor:
    def __init__(self, anthropic_api_key: str, openai_api_key: str, embeddings: Any, vector_store: Optional[Any] = None):
        """Initialize the VideoProcessor."""
        self.anthropic_api_key = SecretStr(anthropic_api_key)
        self.openai_api_key = openai_api_key
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.driver: Optional[WebDriver] = None
        self.llm = ChatAnthropic(
            api_key=self.anthropic_api_key,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            timeout=30,
            stop=None
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self._cleanup_handlers = []

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def cleanup(self):
        """Clean up all resources."""
        try:
            # Clean up WebDriver
            self._quit_driver()
            
            # Clean up vector store if it exists
            if self.vector_store is not None:
                try:
                    self.vector_store.persist()
                except Exception:
                    pass  # Ignore persistence errors
            
            # Run any registered cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception:
                    pass  # Ignore individual handler errors
            
            # Clear references
            self.driver = None
            self.vector_store = None
            self.llm = None
            self.openai_client = None
            self._cleanup_handlers = []
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")

    def register_cleanup_handler(self, handler: Callable[[], None]):
        """Register a cleanup handler to be called during cleanup."""
        self._cleanup_handlers.append(handler)

    def process_video(self, video_url: str, auth_token: Optional[str] = None,
                     playback_speed: Optional[float] = None,
                     duration_limit: Optional[str] = None,
                     time_window: Optional[tuple[str, str]] = None,
                     progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process a video and generate structured notes.
        
        Args:
            video_url: URL of the video to process
            auth_token: Optional authentication token
            playback_speed: Optional playback speed multiplier
            duration_limit: Optional duration limit in HH:MM:SS format
            time_window: Optional tuple of (start_time, end_time) in HH:MM:SS format
            progress_callback: Optional callback for progress updates
        """
        try:
            # Initialize WebDriver if needed for timing features
            if playback_speed or duration_limit or time_window:
                self._initialize_driver()
                if self.driver is None:
                    raise ValueError("Failed to initialize WebDriver")
                
                # Navigate to video URL
                self.driver.get(video_url)
                time.sleep(2)  # Wait for video to load
                
                # Set up timing features
                if time_window:
                    start_time, _ = time_window
                    self._set_video_position(start_time)
                if duration_limit:
                    self._set_duration_limit(duration_limit)
                if playback_speed:
                    self._set_playback_speed(playback_speed)
            
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            
            # Get video transcript with timing information
            if progress_callback:
                progress_callback(10, "Extracting video transcript")
            transcript_data = self._get_transcript(video_id, auth_token)
            
            # Filter transcript data based on time window if specified
            if time_window:
                start_time, end_time = time_window
                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)
                transcript_data = [
                    entry for entry in transcript_data
                    if start_seconds <= entry['start'] <= end_seconds
                ]
            
            if not transcript_data:
                raise ValueError("No transcript data available for the specified time window")
            
            if progress_callback:
                progress_callback(20, "Processing transcript")
            
            # Split transcript into chunks while preserving timing information
            chunks = []
            current_chunk = []
            current_length = 0
            max_chunk_length = 8000  # Maximum length that won't exceed API limits
            
            for entry in transcript_data:
                entry_length = len(entry['text'])
                if current_length + entry_length > max_chunk_length:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_length = 0
                current_chunk.append(entry)
                current_length += entry_length
            
            if current_chunk:
                chunks.append(current_chunk)
            
            print(f"\nSplit transcript into {len(chunks)} chunks")
            print(f"Average chunk length: {current_length / len(chunks) if chunks else 0:.0f} characters")
            
            if progress_callback:
                progress_callback(30, "Identifying main topics")
            
            # First, identify main topics from the entire transcript
            topics_prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert at identifying key topics in educational content. Analyze this video transcript and identify the main topics discussed.

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{{{
    "main_topics": [
        "topic1",
        "topic2",
        "topic3"
    ]
}}}}

Each topic should be a clear, concise phrase that captures a major theme or concept discussed in the transcript.

Here is the transcript to analyze:
{text}"""
            )

            # Define notes prompt template
            notes_prompt = PromptTemplate(
                input_variables=["text", "main_topics"],
                template="""You are an expert at taking detailed, structured notes from educational content. Analyze this section of the transcript and generate detailed notes that align with the main topics.

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure and formatting:
{
    "notes": [
        {
            "topic": "topic_name",
            "content": "Detailed note content",
            "timestamp": "MM:SS"
        }
    ]
}

FORMATTING RULES:
1. Use ONLY double quotes (") for strings, never single quotes
2. Do not include any comments or trailing commas
3. Do not include any explanatory text before or after the JSON
4. Ensure all property names are in double quotes

Here are the main topics identified for this video:
{main_topics}

Here is the transcript section to analyze:
{text}"""
            )
            
            # Get main topics with retries
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # Get main topics
                    full_text = " ".join(entry['text'] for chunk in chunks for entry in chunk)
                    topics_chain = RunnableSequence(first=topics_prompt, last=self.llm)
                    topics_result = topics_chain.invoke({"text": full_text})
                    
                    try:
                        topics_data = json.loads(str(topics_result.content))
                        main_topics = topics_data["main_topics"]
                        break  # Success, exit retry loop
                    except (json.JSONDecodeError, KeyError) as e:
                        if attempt == max_retries - 1:
                            raise ValueError(f"Failed to parse topics from LLM output: {str(e)}")
                        print(f"Retrying topics extraction (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except Exception as e:
                    error_msg = str(e)
                    error_details = {}
                    
                    # Try to extract detailed error information
                    try:
                        if isinstance(e, (APIError, RequestException)) and hasattr(e, 'response') and e.response is not None:
                            error_details = e.response.json()
                        elif hasattr(e, '__cause__'):
                            error_details = {'cause': str(e.__cause__)}
                        elif hasattr(e, '__context__'):
                            error_details = {'context': str(e.__context__)}
                    except:
                        pass
                    
                    if "rate_limit" in error_msg.lower():
                        if attempt == max_retries - 1:
                            raise ValueError(
                                f"Rate limit exceeded after {max_retries} attempts. "
                                f"Error details: {json.dumps(error_details, indent=2)}"
                            )
                        print(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                        print(f"Error details: {json.dumps(error_details, indent=2)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise ValueError(
                            f"Anthropic API error: {error_msg}. "
                            f"Error details: {json.dumps(error_details, indent=2)}"
                        )
            
            if progress_callback:
                progress_callback(40, "Generating detailed notes")
            
            # Now process each chunk to generate detailed notes
            all_notes = []
            chunk_progress = 50 / len(chunks)
            max_retries = 3
            retry_delay = 2
            
            print(f"\nProcessing {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                chunk_text = " ".join(entry['text'] for entry in chunk)
                chunk_start_time = chunk[0]['start'] if chunk else 0
                
                print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
                
                # Try processing the chunk with retries
                for attempt in range(max_retries):
                    try:
                        # Add delay between chunks to avoid rate limits
                        if i > 0:
                            time.sleep(2)
                        
                        notes_chain = RunnableSequence(first=notes_prompt, last=self.llm)
                        result = notes_chain.invoke({
                            "text": chunk_text,
                            "main_topics": "\n".join(main_topics)
                        })
                        
                        print(f"Got response from LLM for chunk {i+1}")
                        print(f"Raw response length: {len(result.content)}")
                        
                        # Clean and parse JSON
                        cleaned_json = self.clean_json_string(str(result.content))
                        print(f"Cleaned JSON length: {len(cleaned_json)}")
                        
                        # Try to parse the JSON
                        try:
                            data = json.loads(cleaned_json)
                            
                            # Validate structure
                            if not isinstance(data, dict) or 'notes' not in data:
                                raise ValueError("Missing 'notes' array in response")
                            if not isinstance(data['notes'], list):
                                raise ValueError("'notes' is not an array")
                            if not data['notes']:
                                raise ValueError("'notes' array is empty")
                                
                            # If we get here, we have valid JSON
                            all_notes.extend(data['notes'])
                            break  # Success, exit retry loop
                            
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON: {str(e)}")
                        
                    except Exception as e:
                        if "rate_limit_error" in str(e):
                            if attempt == max_retries - 1:
                                raise ValueError(f"Rate limit error after {max_retries} attempts: {str(e)}")
                            print(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Failed to process chunk {i+1}/{len(chunks)} after {max_retries} attempts: {str(e)}")
                            print(f"Retrying chunk {i+1}/{len(chunks)} (attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                
                if progress_callback:
                    progress_callback(50 + int((i + 1) * chunk_progress), f"Processing chunk {i+1}/{len(chunks)}")
            
            print(f"\nTotal notes collected: {len(all_notes)}")
            
            if progress_callback:
                progress_callback(90, "Storing notes in database")
            
            # Store notes in vector store
            if self.vector_store is None:
                self.vector_store = Chroma(
                    collection_name="notes",
                    embedding_function=self.embeddings
                )
            
            # Store each note separately in the vector store
            for note in all_notes:
                note_text = json.dumps(note)
                self.vector_store.add_texts(
                    texts=[note_text],
                    metadatas=[{
                        "topic": note["topic"],
                        "source": video_url,
                        "timestamp": note["timestamp"]
                    }]
                )
            
            # Save notes to JSON file in data/notes directory
            os.makedirs("data/notes", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/notes/video_notes_{timestamp}.json"
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "video_url": video_url,
                    "main_topics": main_topics,
                    "notes": all_notes,
                    "processed_at": datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            if progress_callback:
                progress_callback(100, "Processing complete")
            
            return {
                "main_topics": main_topics,
                "notes": all_notes,
                "output_file": output_file
            }
            
        except Exception as e:
            # Extract detailed error information
            error_details = {}
            try:
                if isinstance(e, (APIError, RequestException)) and hasattr(e, 'response') and e.response is not None:
                    error_details = e.response.json()
                elif hasattr(e, '__cause__'):
                    error_details = {'cause': str(e.__cause__)}
                elif hasattr(e, '__context__'):
                    error_details = {'context': str(e.__context__)}
            except:
                pass
            
            error_msg = (
                f"Failed to process video: {str(e)}. "
                f"Error details: {json.dumps(error_details, indent=2)}"
            )
            raise Exception(error_msg)
        finally:
            # Clean up resources
            self.cleanup()
        
    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        # Simple extraction - could be made more robust
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        raise ValueError("Invalid YouTube URL")
        
    def _get_transcript(self, video_id: str, auth_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get video transcript with timing information."""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['en'])  # Try to get English transcript
            # Convert FetchedTranscript to list of dictionaries
            transcript_data = [
                {
                    'text': entry.text,
                    'start': entry.start,
                    'duration': entry.duration
                }
                for entry in transcript.fetch()
            ]
            return transcript_data
        except Exception as e:
            if "No transcript available" in str(e):
                raise ValueError("No transcript available for this video")
            raise ValueError(f"Failed to get transcript: {str(e)}")
        
    def get_notes_by_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Retrieve notes for a specific topic."""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search(
            query=topic,
            k=5  # Return top 5 most relevant results
        )
        
        notes: List[Dict[str, Any]] = []
        for doc in results:
            metadata = doc.metadata or {}
            notes.append({
                "content": doc.page_content,
                "source": metadata.get("source", "unknown"),
                "timestamp": metadata.get("timestamp", "unknown")
            })
        return notes
        
    def merge_notes(self, topics: List[str]) -> Dict:
        """Merge notes from multiple topics into a cohesive summary."""
        # Get notes for all topics
        all_notes = []
        for topic in topics:
            topic_notes = self.get_notes_by_topic(topic)
            all_notes.extend(topic_notes)
        
        if not all_notes:
            return {"summary": "No notes found for the specified topics."}
        
        # Create a summary using the LLM
        prompt = PromptTemplate(
            input_variables=["notes"],
            template="""
            You are an expert at synthesizing information. Given the following notes from multiple topics,
            please create a cohesive summary that connects the key ideas and highlights important relationships.
            
            Notes: {notes}
            
            Summary:
            """
        )
        
        chain = RunnableSequence(first=prompt, last=self.llm)
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        }

    def _initialize_driver(self) -> None:
        """Initialize the Chrome WebDriver."""
        if self.driver is None:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)

    def _quit_driver(self) -> None:
        """Quit the Chrome WebDriver."""
        if self.driver is not None:
            try:
                self.driver.quit()
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                self.driver = None

    def _handle_authentication(self, username: str, password: str) -> None:
        """Handle login if required."""
        if self.driver is None:
            self._initialize_driver()
        
        if self.driver is None:  # Double-check after setup
            raise ValueError("Failed to initialize WebDriver")
            
        driver = self.driver  # Type checker knows it's not None here
        try:
            username_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            password_field = driver.find_element(By.NAME, "password")
            
            username_field.send_keys(username)
            password_field.send_keys(password)
            
            login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for login to complete
            time.sleep(2)
        except TimeoutException:
            raise Exception("Could not find login form or login failed")

    def _set_playback_speed(self, speed: float):
        """Set the video playback speed."""
        # This is a generic implementation - specific sites may need custom handling
        if self.driver is None:
            self._initialize_driver()
        if self.driver is None:
            raise ValueError("Failed to initialize WebDriver")
            
        try:
            # Try to find and click the settings button
            settings_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Settings']"))
            )
            settings_button.click()
            
            # Try to find and click the playback speed option
            speed_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{speed}x')]"))
            )
            speed_button.click()
        except TimeoutException:
            print(f"Warning: Could not set playback speed to {speed}x")

    def _handle_api_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Extract detailed error information from API exceptions."""
        error_details: Dict[str, Any] = {
            'message': str(error),
            'type': error.__class__.__name__,
            'context': context
        }
        
        try:
            if isinstance(error, APIError):
                if error.response is not None:
                    error_details['response'] = error.response.json()
                if error.status_code is not None:
                    error_details['status_code'] = error.status_code
            elif isinstance(error, RequestException):
                if error.response is not None:
                    error_details['response'] = error.response.json()
                    error_details['status_code'] = error.response.status_code
            if hasattr(error, '__cause__'):
                error_details['cause'] = str(error.__cause__)
            if hasattr(error, '__context__'):
                error_details['error_context'] = str(error.__context__)
        except:
            pass
        
        return error_details

    def _handle_rate_limit(self, error: Exception, attempt: int, max_retries: int, retry_delay: int) -> None:
        """Handle rate limit errors with exponential backoff."""
        error_details = self._handle_api_error(error, "Rate limit error")
        
        if attempt == max_retries - 1:
            raise ValueError(
                f"Rate limit exceeded after {max_retries} attempts. "
                f"Error details: {json.dumps(error_details, indent=2)}"
            )
        
        print(f"Rate limit hit, waiting {retry_delay} seconds before retry...")
        print(f"Error details: {json.dumps(error_details, indent=2)}")
        time.sleep(retry_delay)

    def clean_json_string(self, json_str: str) -> str:
        """Clean and validate JSON string."""
        if not isinstance(json_str, str):
            raise ValueError("Input must be a string")

        # Remove code block markers if present
        if json_str.startswith('```') and json_str.endswith('```'):
            lines = json_str.split('\n')
            json_str = '\n'.join(lines[1:-1])
        if json_str.startswith('```json'):
            json_str = json_str[7:]

        # Remove any leading/trailing whitespace
        json_str = json_str.strip()

        try:
            # Try to parse as is first
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If that fails, try to clean up the JSON
            # Replace single quotes with double quotes
            json_str = json_str.replace("'", '"')
            
            # Remove any comments
            json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.S)
            
            # Add quotes to unquoted property names
            json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
            
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # Fix missing quotes around string values
            json_str = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])([,}\]])', r':"\1"\2', json_str)

            # Try to parse again
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                # If still failing, try one more time with a more aggressive cleaning
                try:
                    # Remove all whitespace and newlines
                    json_str = ''.join(json_str.split())
                    # Add minimal formatting
                    json_str = json_str.replace('{', '{\n  ').replace('}', '\n}').replace(',', ',\n  ')
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON after cleaning: {str(e)}")

    def get_llm_response(self, chunk: str) -> str:
        """Get response from LLM for a chunk of text."""
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized")
            
        prompt = f"""Given the following video transcript chunk, generate detailed notes in JSON format.
Focus on key points, main ideas, and important details.
The notes should be clear, concise, and well-organized.

Transcript chunk:
{chunk}

Return ONLY a JSON object with this exact structure:
{{
    "notes": [
        {{
            "topic": "Brief topic description",
            "content": "Detailed content of the note"
        }}
    ]
}}

IMPORTANT: 
1. Do not include any explanations or text outside the JSON object
2. Ensure all JSON is properly formatted with no trailing commas
3. Use double quotes for all strings
4. Keep responses concise and focused
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates detailed notes from video transcripts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096
            )
            
            # Get the response content
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI")
                
            content = content.strip()
            
            # Basic validation before returning
            if not content.startswith('{') or not content.endswith('}'):
                raise ValueError("Response is not a valid JSON object")
                
            return content
        except Exception as e:
            if "rate_limit_error" in str(e):
                # If we hit a rate limit, wait and retry
                time.sleep(60)  # Wait 60 seconds before retrying
                return self.get_llm_response(chunk)  # Retry the request
            raise ValueError(f"Error getting LLM response: {str(e)}")

    def process_chunk(self, chunk: str, chunk_number: int, total_chunks: int) -> Dict[str, Any]:
        """Process a single chunk of text."""
        max_retries = 3
        last_error = None
        retry_delay = 2  # Start with 2 second delay
        
        for attempt in range(max_retries):
            try:
                # Get response from LLM
                response = self.get_llm_response(chunk)
                print(f"\nProcessing chunk {chunk_number}/{total_chunks} (attempt {attempt + 1})")
                print(f"Raw response length: {len(response)}")
                
                # Clean and parse JSON
                cleaned_json = self.clean_json_string(response)
                print(f"Cleaned JSON length: {len(cleaned_json)}")
                
                # Try to parse the JSON
                try:
                    data = json.loads(cleaned_json)
                    
                    # Validate structure
                    if not isinstance(data, dict) or 'notes' not in data:
                        raise ValueError("Missing 'notes' array in response")
                    if not isinstance(data['notes'], list):
                        raise ValueError("'notes' is not an array")
                    if not data['notes']:
                        raise ValueError("'notes' array is empty")
                    
                    # Add any missing fields
                    for note in data['notes']:
                        note.setdefault('key_points', [])
                        note.setdefault('examples', [])
                        note.setdefault('source', 'video transcript')
                        note.setdefault('timestamp', datetime.now().isoformat())
                        note.setdefault('time_elapsed', '0')
                        
                    # If we get here, we have valid JSON
                    return data
                    
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {str(e)}")
                
            except Exception as e:
                last_error = str(e)
                if "rate_limit_error" in last_error:
                    # If we hit a rate limit, wait longer before retrying
                    retry_delay = 60  # Wait 60 seconds for rate limits
                if attempt < max_retries - 1:
                    print(f"\nError on attempt {attempt + 1}: {last_error}")
                    print(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    error_msg = f"Failed to process chunk {chunk_number}/{total_chunks} after {max_retries} attempts"
                    error_msg += f"\nLast error: {last_error}"
                    raise ValueError(error_msg)
        
        # If we get here, all retries failed
        return {"notes": []}

    def _time_to_seconds(self, time_str: str) -> float:
        """Convert time string in HH:MM:SS format to seconds."""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
            else:
                raise ValueError(f"Invalid time format: {time_str}")
        except ValueError as e:
            raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS or MM:SS")

    def _set_video_position(self, time_str: str) -> None:
        """Set video position to specified time."""
        if self.driver is None:
            self._initialize_driver()
        if self.driver is None:
            raise ValueError("Failed to initialize WebDriver")
            
        try:
            # Try to find and click the video player
            video_player = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            
            # Convert time to seconds and seek
            seconds = self._time_to_seconds(time_str)
            self.driver.execute_script(f"arguments[0].currentTime = {seconds};", video_player)
            
            # Wait for seek to complete
            time.sleep(1)
        except TimeoutException:
            print(f"Warning: Could not set video position to {time_str}")

    def _set_duration_limit(self, duration: str) -> None:
        """Set up duration limit for video playback."""
        if self.driver is None:
            self._initialize_driver()
        if self.driver is None:
            raise ValueError("Failed to initialize WebDriver")
            
        try:
            # Convert duration to seconds
            seconds = self._time_to_seconds(duration)
            
            # Add event listener to stop video after duration
            script = f"""
            var video = document.querySelector('video');
            if (video) {{
                var startTime = video.currentTime;
                var checkTime = setInterval(function() {{
                    if (video.currentTime - startTime >= {seconds}) {{
                        video.pause();
                        clearInterval(checkTime);
                    }}
                }}, 1000);
            }}
            """
            self.driver.execute_script(script)
        except Exception as e:
            print(f"Warning: Could not set duration limit: {str(e)}") 