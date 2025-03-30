import os
from typing import List, Dict, Optional, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from datetime import datetime
import json
import time
from pydantic import SecretStr

class TextProcessor:
    def __init__(self, anthropic_api_key: str):
        """Initialize the text processor with Anthropic API key."""
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
            timeout=30,
            stop=None,
            temperature=0.7
        )
        self.vector_store = None

    def process_text(self, text: str,
                    progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict:
        """Process text and generate structured notes."""
        try:
            if progress_callback:
                progress_callback(10, "Splitting text into chunks")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            if progress_callback:
                progress_callback(30, "Identifying main topics")
            
            # First, identify main topics from the entire text
            topics_prompt = PromptTemplate(
                input_variables=["text"],
                template="""You are an expert at identifying key topics in educational content. Analyze this text and identify the main topics discussed.

IMPORTANT: You MUST respond with ONLY a valid JSON object and NOTHING else - no explanations, no other text.
The JSON object MUST follow this EXACT structure:
{{{{
    "main_topics": [
        "topic1",
        "topic2",
        "topic3"
    ]
}}}}

Each topic should be a clear, concise phrase that captures a major theme or concept discussed in the text.

Here is the text to analyze:
{text}"""
            )
            
            # Get main topics
            full_text = " ".join(chunks)
            topics_chain = topics_prompt | self.llm
            topics_result = topics_chain.invoke({"text": full_text})
            
            try:
                topics_data = json.loads(str(topics_result.content))
                main_topics = topics_data["main_topics"]
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"Failed to parse topics from LLM output: {str(e)}")
            
            if progress_callback:
                progress_callback(40, "Generating detailed notes")
            
            # Now process each chunk to generate detailed notes
            notes_prompt = PromptTemplate(
                input_variables=["text", "main_topics"],
                template="""You are an expert note-taker. Your task is to analyze this section of the text and create structured notes.

The main topics identified in this text are:
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
            "source": "text",
            "timestamp": "current datetime"
        }}}}
    ]
}}}}

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
                        notes_chain = notes_prompt | self.llm
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
                                "source": "text input",
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
                        "source": "text input",
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
            raise Exception(f"Failed to process text: {str(e)}")
        
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
        
        chain = prompt | self.llm
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        }

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks for processing."""
        return self.text_splitter.split_text(text)

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts."""
        return self.embeddings.embed_documents(texts)

    def create_vector_store(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> Chroma:
        """Create a vector store from texts and optional metadata."""
        return Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

    def generate_summary(self, text: str) -> str:
        """Generate a summary of the text using LLM."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Please provide a concise summary of the following text:
            
            {text}
            
            Summary:"""
        )
        
        # Use the new runnable pattern instead of LLMChain
        chain = prompt | self.llm
        return str(chain.invoke({"text": text}).content)

    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from the text using LLM."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Please extract the key points from the following text:
            
            {text}
            
            Key Points:"""
        )
        
        # Use the new runnable pattern instead of LLMChain
        chain = prompt | self.llm
        response = str(chain.invoke({"text": text}).content)
        return [point.strip() for point in response.split('\n') if point.strip()]

    def generate_structured_notes(self, text: str) -> Dict:
        """Generate structured notes from text using LLM."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Please create structured notes from the following text:
            
            {text}
            
            Create a JSON structure with main topics and subtopics. Format:
            {
                "main_topics": [
                    {
                        "topic": "Main Topic",
                        "subtopics": [
                            {
                                "subtopic": "Subtopic",
                                "points": ["Point 1", "Point 2"]
                            }
                        ]
                    }
                ]
            }"""
        )
        
        # Use the new runnable pattern instead of LLMChain
        chain = prompt | self.llm
        response = str(chain.invoke({"text": text}).content)
        # Note: In a real implementation, you'd want to properly parse the JSON response
        # and handle potential errors. This is a simplified version.
        return eval(response) 