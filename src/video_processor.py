import os
from typing import Dict, List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from datetime import datetime

class VideoProcessor:
    def __init__(self, anthropic_api_key: str):
        """Initialize the video processor with Anthropic API key."""
        self.anthropic_api_key = anthropic_api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.llm = ChatAnthropic(
            anthropic_api_key=anthropic_api_key,
            model="claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=4096
        )
        self.vector_store = None
        self.driver = None

    def process_video(self, video_url: str, auth_token: Optional[str] = None, playback_speed: float = 2.0) -> Dict:
        """Process a video and generate structured notes.
        
        Args:
            video_url: URL of the YouTube video
            auth_token: Optional authentication token for private videos
            playback_speed: Desired playback speed (default: 2.0x)
        """
        try:
            # Set up browser for video playback
            self._setup_driver()
            self._navigate_to_video(video_url)
            self._set_playback_speed(playback_speed)
            
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            
            # Get video transcript with timing information
            transcript_data = self._get_transcript(video_id, auth_token)
            
            # Split transcript into chunks while preserving timing information
            chunks = []
            current_chunk = []
            current_length = 0
            
            for entry in transcript_data:
                if current_length + len(entry['text']) > 1000:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_length = 0
                current_chunk.append(entry)
                current_length += len(entry['text'])
            
            if current_chunk:
                chunks.append(current_chunk)
            
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
            
            # Get main topics
            full_text = " ".join(entry['text'] for chunk in chunks for entry in chunk)
            topics_chain = topics_prompt | self.llm
            topics_result = topics_chain.invoke({"text": full_text})
            
            try:
                topics_data = json.loads(topics_result.content)
                main_topics = topics_data["main_topics"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse topics from LLM output: {str(e)}")
            
            # Now process each chunk to generate detailed notes
            notes_prompt = PromptTemplate(
                input_variables=["text", "main_topics"],
                template="""You are an expert note-taker. Your task is to analyze this section of the video transcript and create structured notes.

The main topics identified in this video are:
{main_topics}

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{{{
    "notes": [
        {{{{
            "topic": "one of the main topics listed above",
            "content": "detailed notes about the topic from this section",
            "key_points": ["point1", "point2", "point3"],
            "examples": ["example1", "example2"],
            "source": "video transcript",
            "timestamp": "current datetime",
            "time_elapsed": "time in seconds when this topic appears in the video"
        }}}}
    ]
}}}}

For each note:
1. The topic MUST match one of the main topics listed above
2. Include specific details and insights from this section
3. Extract key points as a list
4. Include relevant examples or analogies if present
5. Focus on accuracy and clarity
6. Set time_elapsed to the start time of the first relevant segment in this chunk

Here is the transcript section to analyze:
{text}"""
            )
            
            # Process each chunk and collect notes
            all_notes = []
            for chunk in chunks:
                chunk_text = " ".join(entry['text'] for entry in chunk)
                chunk_start_time = chunk[0]['start'] if chunk else 0
                
                notes_chain = notes_prompt | self.llm
                result = notes_chain.invoke({
                    "text": chunk_text,
                    "main_topics": "\n".join(main_topics)
                })
                
                try:
                    chunk_notes = json.loads(result.content)
                    # Add timing information to each note
                    for note in chunk_notes.get("notes", []):
                        note["time_elapsed"] = chunk_start_time
                    all_notes.extend(chunk_notes.get("notes", []))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse notes from chunk: {str(e)}")
                    continue
            
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
            
            return {
                "main_topics": main_topics,
                "notes": all_notes
            }
        finally:
            self._cleanup()

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        # Simple extraction - could be made more robust
        if "youtu.be" in url:
            return url.split("/")[-1]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        raise ValueError("Invalid YouTube URL")
        
    def _get_transcript(self, video_id: str, auth_token: Optional[str] = None) -> Dict:
        """Get video transcript using YouTube Transcript API."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            # Format transcript with timing information
            formatted_transcript = []
            for entry in transcript_list:
                formatted_transcript.append({
                    'text': entry['text'],
                    'start': entry['start'],
                    'duration': entry['duration']
                })
            return formatted_transcript
        except Exception as e:
            raise ValueError(f"Failed to get transcript: {str(e)}")
        
    def get_notes_by_topic(self, topic: str) -> List[Dict]:
        """Retrieve notes for a specific topic."""
        if self.vector_store is None:
            return []
        
        results = self.vector_store.similarity_search(
            query=topic,
            k=5  # Return top 5 most relevant results
        )
        
        return [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "timestamp": doc.metadata.get("timestamp", "unknown")
            }
            for doc in results
        ]
        
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
        
        # Use the new runnable pattern instead of LLMChain
        chain = prompt | self.llm
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        }

    def _setup_driver(self):
        """Set up the Selenium WebDriver with improved options."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-infobars')
        options.add_argument('--mute-audio')  # Mute audio by default
        self.driver = webdriver.Chrome(options=options)

    def _navigate_to_video(self, url: str):
        """Navigate to the video URL."""
        self.driver.get(url)
        # Wait for video player to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "video"))
        )

    def _handle_authentication(self, username: str, password: str):
        """Handle login if required."""
        # This is a generic implementation - specific sites may need custom handling
        try:
            username_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "username"))
            )
            password_field = self.driver.find_element(By.NAME, "password")
            
            username_field.send_keys(username)
            password_field.send_keys(password)
            
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for login to complete
            time.sleep(2)
        except TimeoutException:
            raise Exception("Could not find login form or login failed")

    def _set_playback_speed(self, speed: float):
        """Set the video playback speed with improved reliability."""
        try:
            # Wait for video player to be ready
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "video"))
            )
            
            # Try multiple methods to set playback speed
            methods = [
                self._set_speed_via_settings,
                self._set_speed_via_keyboard,
                self._set_speed_via_javascript
            ]
            
            for method in methods:
                try:
                    if method(speed):
                        print(f"Successfully set playback speed to {speed}x")
                        return
                except Exception as e:
                    print(f"Warning: Failed to set speed via {method.__name__}: {str(e)}")
                    continue
            
            print(f"Warning: Could not set playback speed to {speed}x using any method")
            
        except Exception as e:
            print(f"Warning: Error setting playback speed: {str(e)}")

    def _set_speed_via_settings(self, speed: float) -> bool:
        """Set playback speed using the settings menu."""
        try:
            # Click settings button
            settings_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Settings']"))
            )
            settings_button.click()
            
            # Click playback speed option
            speed_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{speed}x')]"))
            )
            speed_button.click()
            return True
        except:
            return False

    def _set_speed_via_keyboard(self, speed: float) -> bool:
        """Set playback speed using keyboard shortcuts."""
        try:
            # Focus on video player
            video = self.driver.find_element(By.TAG_NAME, "video")
            video.click()
            
            # Send keyboard shortcut (Shift + > for faster)
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.common.action_chains import ActionChains
            
            actions = ActionChains(self.driver)
            actions.key_down(Keys.SHIFT).send_keys('>').key_up(Keys.SHIFT).perform()
            return True
        except:
            return False

    def _set_speed_via_javascript(self, speed: float) -> bool:
        """Set playback speed using JavaScript."""
        try:
            script = f"""
            var video = document.querySelector('video');
            if (video) {{
                video.playbackRate = {speed};
                return true;
            }}
            return false;
            """
            return self.driver.execute_script(script)
        except:
            return False

    def _cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit() 