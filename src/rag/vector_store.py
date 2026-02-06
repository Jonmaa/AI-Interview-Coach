"""
Vector Store module using FAISS (free, local).
Stores and retrieves document embeddings for RAG.
No paid APIs required - uses local HuggingFace embeddings.
FAISS is fully compatible with Python 3.14.
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embeddings import get_embeddings


# Default path for the vector store
FAISS_PERSIST_DIR = Path(__file__).parent.parent.parent / "data" / "faiss_db"


class VectorStore:
    """
    Manages the FAISS vector database for storing and retrieving documents.
    
    Usage:
        store = VectorStore()
        store.add_documents([Document(page_content="...", metadata={...})])
        results = store.search("your query", k=5)
    """
    
    def __init__(self, index_name: str = "interview_coach", persist_directory: str = None):
        """
        Initialize the vector store.
        
        Args:
            index_name: Name of the FAISS index
            persist_directory: Where to store the database (optional)
        """
        self.index_name = index_name
        self.persist_directory = persist_directory or str(FAISS_PERSIST_DIR)
        self.embeddings = get_embeddings()
        
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Try to load existing index, otherwise start empty
        index_path = Path(self.persist_directory) / f"{index_name}.faiss"
        if index_path.exists():
            self.vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )
            print(f"✅ Vector store loaded from disk: {index_name}")
        else:
            self.vectorstore = None  # Created on first add
            print(f"✅ Vector store initialized (empty): {index_name}")
    
    def _save(self) -> None:
        """Persist the FAISS index to disk."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.persist_directory, index_name=self.index_name)
    
    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            print("⚠️ No documents to add")
            return
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        
        self._save()
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def add_texts(self, texts: list[str], metadatas: list[dict] = None) -> None:
        """
        Add raw texts to the vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts for each text
        """
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            self.vectorstore.add_texts(texts, metadatas=metadatas)
        
        self._save()
        print(f"✅ Added {len(texts)} texts to vector store")
    
    def search(self, query: str, k: int = 5) -> list[Document]:
        """
        Search for similar documents.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
        """
        Search for similar documents with similarity scores.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def get_retriever(self, k: int = 5):
        """
        Get a retriever for use with LangChain chains.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            A retriever object
        """
        if self.vectorstore is None:
            # Create a minimal store so we can return a retriever
            self.add_texts(["placeholder"])
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vectorstore = None
        # Remove persisted files
        for ext in (".faiss", ".pkl"):
            p = Path(self.persist_directory) / f"{self.index_name}{ext}"
            p.unlink(missing_ok=True)
        print("🧹 Vector store cleared")


if __name__ == "__main__":
    print("=" * 50)
    print("VECTOR STORE TEST (FAISS)")
    print("=" * 50)
    
    # Create a test vector store
    store = VectorStore(index_name="test_collection")
    
    # Add some test documents
    test_docs = [
        "LangChain is a framework for building LLM applications with agents and tools.",
        "HuggingFace embeddings convert text into numerical vectors for semantic search locally.",
        "RAG (Retrieval Augmented Generation) combines search with LLM generation.",
        "Python is the most popular language for AI and machine learning development.",
        "Function calling allows LLMs to invoke external tools and APIs.",
    ]
    
    store.add_texts(test_docs, metadatas=[{"source": f"test_{i}"} for i in range(len(test_docs))])
    
    # Test search
    query = "How do I build an agent with LangChain?"
    print(f"\n🔍 Searching for: '{query}'")
    
    results = store.search(query, k=3)
    print(f"\n📄 Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"   {i}. {doc.page_content[:80]}...")
    
    print("\n✅ Test completed!")
