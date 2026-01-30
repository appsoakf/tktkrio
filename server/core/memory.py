from abc import ABC, abstractmethod
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from server.core.config import settings

class MemoryInterface(ABC):
    @abstractmethod
    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """Add a memory fragment to the vector database."""
        pass

    @abstractmethod
    async def query_memory(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant memory fragments based on query."""
        pass

class ChromaMemoryService(MemoryInterface):
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        # Assuming we use a default embedding function provided by Chroma or configured externally
        # For a production setup, you might want to explicitly set the embedding function here
        self.collection = self.client.get_or_create_collection(name="long_term_memory")

    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        if metadata is None:
            metadata = {}
        
        # Simple ID generation based on timestamp or hash could be better
        import uuid
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )

    async def query_memory(self, query: str, n_results: int = 3) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # Flatten the list of documents
        if results and results['documents']:
            return results['documents'][0]
        return []
