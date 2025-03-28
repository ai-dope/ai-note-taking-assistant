import os
from typing import List, Dict, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

class VectorStore:
    def __init__(self, openai_api_key: str, persist_directory: str = "data/vector_store"):
        """Initialize the vector store with OpenAI API key and persistence directory."""
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store with optional metadata."""
        return self.vector_store.add_texts(texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """Search for similar texts using a query string."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete_collection(self):
        """Delete the current collection."""
        self.vector_store.delete_collection()

    def persist(self):
        """Persist the vector store to disk."""
        self.vector_store.persist()

    def get_all_texts(self) -> List[str]:
        """Get all texts stored in the vector store."""
        return self.vector_store.get()["documents"] 