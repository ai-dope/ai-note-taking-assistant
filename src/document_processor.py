import os
import magic
from typing import Dict, List, Optional
from docx import Document
from PyPDF2 import PdfReader
from .text_processor import TextProcessor
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from datetime import datetime

class DocumentProcessor:
    def __init__(self, anthropic_api_key: str):
        """Initialize the document processor with Anthropic API key."""
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

    def process_document(self, file_path: str) -> Dict:
        """Process a document and generate structured notes.

        Args:
            file_path (str): Path to the document file.

        Returns:
            Dict: A dictionary containing the structured notes.
        """
        content = self._extract_content(file_path)
        chunks = self.text_splitter.split_text(content)
        
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Please analyze the following text and generate structured notes in JSON format.
            The notes should include main topics and detailed points for each topic.

            Text: {text}

            Generate a JSON response in the following format:
            {{
                "main_topics": [
                    {{
                        "topic": "Topic 1",
                        "notes": [
                            "Detailed point 1",
                            "Detailed point 2"
                        ]
                    }},
                    {{
                        "topic": "Topic 2",
                        "notes": [
                            "Detailed point 1",
                            "Detailed point 2"
                        ]
                    }}
                ]
            }}
            """
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"text": "\n".join(chunks)})
        
        # Parse the result into a Python dictionary
        try:
            import json
            notes = json.loads(result.content)
        except json.JSONDecodeError:
            # If the result is not valid JSON, try to extract the JSON part
            import re
            json_match = re.search(r'\{.*\}', result.content, re.DOTALL)
            if json_match:
                try:
                    notes = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse LLM response as JSON")
            else:
                raise ValueError("Failed to find JSON in LLM response")
        
        # Store the notes in the vector store
        if self.vector_store is None:
            self.vector_store = Chroma(
                collection_name="notes",
                embedding_function=self.embeddings
            )
        
        # Add the notes to the vector store
        self.vector_store.add_texts(
            texts=[json.dumps(notes)],
            metadatas=[{"source": os.path.basename(file_path), "timestamp": datetime.now().isoformat()}]
        )
        
        return notes

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
        
        chain = prompt | self.llm
        summary = chain.invoke({"notes": str(all_notes)}).content
        
        return {
            "summary": summary,
            "source_topics": topics,
            "timestamp": datetime.now().isoformat()
        } 