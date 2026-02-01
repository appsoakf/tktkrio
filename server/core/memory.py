from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from server.core.config import settings

logger = logging.getLogger(__name__)

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
    """
    Long-term memory service using ChromaDB for vector storage and RAG.

    Uses BGE-M3 or sentence-transformers for embeddings to retrieve
    semantically similar past conversations.
    """

    def __init__(self):
        logger.info(f"Initializing ChromaDB at: {settings.CHROMA_DB_PATH}")

        try:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

            # Use HuggingFace embedding function with BGE-M3 model
            # This provides better multilingual support (Chinese + English)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL_NAME,
                device=settings.DEVICE
            )

            self.collection = self.client.get_or_create_collection(
                name="long_term_memory",
                embedding_function=self.embedding_function,
                metadata={"description": "AI Companion long-term conversation memory"}
            )

            logger.info(f"✓ ChromaDB initialized with {self.collection.count()} existing memories")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        """
        Add a conversation memory to the vector database.

        Args:
            text: The conversation text to store
            metadata: Optional metadata (timestamp, emotion, etc.)
        """
        if not text or not text.strip():
            return

        if metadata is None:
            metadata = {}

        # Add timestamp if not provided
        if "timestamp" not in metadata:
            from datetime import datetime
            metadata["timestamp"] = datetime.now().isoformat()

        doc_id = str(uuid.uuid4())

        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.debug(f"Added memory: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    async def query_memory(self, query: str, n_results: int = 3) -> List[str]:
        """
        Retrieve relevant memory fragments using semantic similarity.

        Args:
            query: Query text to search for similar memories
            n_results: Maximum number of results to return

        Returns:
            List of relevant memory text strings
        """
        if not query or not query.strip():
            return []

        try:
            # Don't query more results than stored
            count = self.collection.count()
            if count == 0:
                return []

            actual_n = min(n_results, count)

            results = self.collection.query(
                query_texts=[query],
                n_results=actual_n
            )

            if results and results['documents']:
                return results['documents'][0]
            return []

        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return []
