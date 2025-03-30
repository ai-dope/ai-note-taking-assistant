import os
import magic
import json
import time
from typing import Dict, List, Optional, Callable, Union, Any
from docx import Document
from PyPDF2 import PdfReader
from .text_processor import TextProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableSequence
from datetime import datetime
from pydantic import SecretStr

def safe_json_loads(s: Union[str, bytes, bytearray, Any]) -> Any:
    """Safely load JSON string with proper type handling."""
    if isinstance(s, (bytes, bytearray)):
        s = s.decode('utf-8')
    elif not isinstance(s, str):
        s = str(s)
    return json.loads(s)

class DocumentProcessor:
    def __init__(self, anthropic_api_key: str):
        """Initialize the document processor with Anthropic API key."""
        self.anthropic_api_key = SecretStr(anthropic_api_key)
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
            api_key=self.anthropic_api_key,
            model_name="claude-3-haiku-20240307",
            temperature=0.7,
            timeout=30,
            stop=None
        )
        self.vector_store = None
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
            self.vector_store = None
            self.llm = None
            self._cleanup_handlers = []
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")

    def register_cleanup_handler(self, handler: Callable[[], None]):
        """Register a cleanup handler to be called during cleanup."""
        self._cleanup_handlers.append(handler)

    def process_document(self, file_path: str,
                        progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process a document and generate structured notes."""
        try:
            # Read document content
            if progress_callback:
                progress_callback(10, "Reading document")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            if progress_callback:
                progress_callback(20, "Splitting document into chunks")
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            if progress_callback:
                progress_callback(30, "Identifying main topics")
            
            # First, identify main topics from the entire document
            topics_prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert at identifying key topics in educational content. Analyze this document and identify the main topics discussed.

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{
    "main_topics": [
        "topic1",
        "topic2",
        "topic3"
    ]
}}

Each topic should be a clear, concise phrase that captures a major theme or concept discussed in the document.

Here is the document to analyze:
{text}"""
            )
            
            # Get main topics
            full_text = " ".join(chunks)
            topics_chain = RunnableSequence(first=topics_prompt, last=self.llm)
            topics_result = topics_chain.invoke({"text": full_text})
            
            try:
                topics_data = safe_json_loads(topics_result.content)
                main_topics = topics_data["main_topics"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse topics from LLM output: {str(e)}")
            
            if progress_callback:
                progress_callback(40, "Generating detailed notes")
            
            # Now process each chunk to generate detailed notes
            notes_prompt = PromptTemplate(
                input_variables=["text", "main_topics"],
                template="""You are an expert note-taker. Your task is to analyze this section of the document and create structured notes.

The main topics identified in this document are:
{main_topics}

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{
    "notes": [
        {{
            "topic": "one of the main topics listed above",
            "content": "detailed notes about the topic from this section",
            "key_points": ["point1", "point2", "point3"],
            "examples": ["example1", "example2"],
            "source": "document",
            "timestamp": "current datetime"
        }}
    ]
}}

For each note:
1. The topic MUST match one of the main topics listed above
2. Include specific details and insights from this section
3. Extract key points as a list
4. Include relevant examples or analogies if present
5. Focus on accuracy and clarity

Here is the document section to analyze:
{text}"""
            )
            
            # Process each chunk and collect notes
            all_notes = []
            chunk_progress = 50 / len(chunks)
            max_retries = 3
            
            for i, chunk in enumerate(chunks):
                # Try processing the chunk with retries
                for retry in range(max_retries):
                    try:
                        notes_chain = RunnableSequence(first=notes_prompt, last=self.llm)
                        result = notes_chain.invoke({
                            "text": chunk,
                            "main_topics": "\n".join(main_topics)
                        })
                        
                        # Clean the response to ensure valid JSON
                        cleaned_content = str(result.content).strip()
                        if cleaned_content.startswith("```json"):
                            cleaned_content = cleaned_content[7:]
                        if cleaned_content.endswith("```"):
                            cleaned_content = cleaned_content[:-3]
                        cleaned_content = cleaned_content.strip()
                        
                        chunk_notes = safe_json_loads(cleaned_content)
                        
                        # Validate the structure
                        if not isinstance(chunk_notes, dict) or "notes" not in chunk_notes:
                            raise ValueError("Invalid response structure")
                        
                        # Ensure all required fields are present
                        for note in chunk_notes.get("notes", []):
                            note.setdefault("key_points", [])
                            note.setdefault("examples", [])
                            note.setdefault("timestamp", datetime.now().isoformat())
                        
                        all_notes.extend(chunk_notes.get("notes", []))
                        break  # Success, exit retry loop
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        if retry == max_retries - 1:  # Last retry
                            print(f"Warning: Failed to parse notes from chunk {i+1}/{len(chunks)} after {max_retries} attempts: {str(e)}")
                            # Add a placeholder note to maintain progress
                            all_notes.append({
                                "topic": "Error",
                                "content": f"Failed to process chunk {i+1}",
                                "key_points": [],
                                "examples": [],
                                "source": "document",
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            print(f"Retrying chunk {i+1}/{len(chunks)} (attempt {retry + 1}/{max_retries})")
                            time.sleep(1)  # Wait before retrying
                
                if progress_callback:
                    progress_callback(50 + int((i + 1) * chunk_progress), f"Processing chunk {i+1}/{len(chunks)}")
            
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
                        "source": file_path,
                        "timestamp": note["timestamp"]
                    }]
                )
            
            if progress_callback:
                progress_callback(100, "Processing complete")
            
            return {
                "main_topics": main_topics,
                "notes": all_notes
            }
            
        except Exception as e:
            raise Exception(f"Failed to process document: {str(e)}")
        finally:
            # Clean up resources
            self.cleanup()

    def process_text(self, text: str, progress_callback: Optional[Callable[[int], None]] = None) -> Dict:
        """Process raw text content and generate structured notes."""
        try:
            # Split text into chunks
            if progress_callback:
                progress_callback(10)
            
            chunks = self.text_splitter.split_text(text)
            
            if progress_callback:
                progress_callback(20)
            
            # First, identify main topics from the entire text
            topics_prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert at identifying key topics in educational content. Analyze this text and identify the main topics discussed.

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{
    "main_topics": [
        "topic1",
        "topic2",
        "topic3"
    ]
}}

Each topic should be a clear, concise phrase that captures a major theme or concept discussed in the text.

Here is the text to analyze:
{text}"""
            )
            
            # Get main topics
            full_text = " ".join(chunks)
            topics_chain = RunnableSequence(first=topics_prompt, last=self.llm)
            topics_result = topics_chain.invoke({"text": full_text})
            
            try:
                topics_data = json.loads(str(topics_result.content))
                main_topics = topics_data["main_topics"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse topics from LLM output: {str(e)}")
            
            if progress_callback:
                progress_callback(30)
            
            # Now process each chunk to generate detailed notes
            notes_prompt = PromptTemplate(
                input_variables=["text", "main_topics"],
                template="""You are an expert note-taker. Your task is to analyze this section of the text and create structured notes.

The main topics identified in this text are:
{main_topics}

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{
    "notes": [
        {
            "topic": "one of the main topics listed above",
            "content": "detailed notes about the topic from this section",
            "key_points": ["point1", "point2", "point3"],
            "examples": ["example1", "example2"],
            "source": "text",
            "timestamp": "current datetime"
        }
    ]
}}

For each note:
1. The topic MUST match one of the main topics listed above
2. Include specific details and insights from this section
3. Extract key points as a list
4. Include relevant examples or analogies if present
5. Focus on accuracy and clarity

Here is the text section to analyze:
{text}"""
            )
            
            # Process each chunk and collect notes
            all_notes = []
            chunk_progress = 50 / len(chunks)
            max_retries = 3
            
            for i, chunk in enumerate(chunks):
                # Try processing the chunk with retries
                for retry in range(max_retries):
                    try:
                        notes_chain = RunnableSequence(first=notes_prompt, last=self.llm)
                        result = notes_chain.invoke({
                            "text": chunk,
                            "main_topics": "\n".join(main_topics)
                        })
                        
                        # Clean the response to ensure valid JSON
                        cleaned_content = str(result.content).strip()
                        if cleaned_content.startswith("```json"):
                            cleaned_content = cleaned_content[7:]
                        if cleaned_content.endswith("```"):
                            cleaned_content = cleaned_content[:-3]
                        cleaned_content = cleaned_content.strip()
                        
                        chunk_notes = json.loads(cleaned_content)
                        
                        # Validate the structure
                        if not isinstance(chunk_notes, dict) or "notes" not in chunk_notes:
                            raise ValueError("Invalid response structure")
                        
                        # Ensure all required fields are present
                        for note in chunk_notes.get("notes", []):
                            note.setdefault("key_points", [])
                            note.setdefault("examples", [])
                            note.setdefault("timestamp", datetime.now().isoformat())
                        
                        all_notes.extend(chunk_notes.get("notes", []))
                        break  # Success, exit retry loop
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        if retry == max_retries - 1:  # Last retry
                            print(f"Warning: Failed to parse notes from chunk {i+1}/{len(chunks)} after {max_retries} attempts: {str(e)}")
                            # Add a placeholder note to maintain progress
                            all_notes.append({
                                "topic": "Error",
                                "content": f"Failed to process chunk {i+1}",
                                "key_points": [],
                                "examples": [],
                                "source": "text",
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            print(f"Retrying chunk {i+1}/{len(chunks)} (attempt {retry + 1}/{max_retries})")
                            time.sleep(1)  # Wait before retrying
                
                if progress_callback:
                    progress_callback(50 + int((i + 1) * chunk_progress))
            
            if progress_callback:
                progress_callback(90)
            
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
                        "source": "text",
                        "timestamp": note["timestamp"]
                    }]
                )
            
            if progress_callback:
                progress_callback(100)
            
            return {
                "main_topics": main_topics,
                "notes": all_notes
            }
            
        except Exception as e:
            raise Exception(f"Failed to process text: {str(e)}")
        finally:
            # Clean up resources
            self.cleanup()

    def _extract_content(self, file_path: str) -> str:
        """Extract text content from a document."""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        
        if file_type == "application/pdf":
            return self._extract_pdf(file_path)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._extract_docx(file_path)
        elif file_type == "text/plain":
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file."""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def _extract_txt(self, file_path: str) -> str:
        """Extract text from a plain text file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

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
        
        chain = RunnableSequence(first=prompt, last=self.llm)
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        } 