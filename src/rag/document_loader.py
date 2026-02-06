"""
Document Loader module.
Loads documents from various formats (PDF, TXT, DOCX) for RAG.
"""

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader,
)


class DocumentLoader:
    """
    Loads and processes documents from various formats.
    
    Usage:
        loader = DocumentLoader()
        docs = loader.load_file("path/to/document.pdf")
        docs = loader.load_directory("path/to/folder")
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks for context continuity
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_file(self, file_path: str) -> list[Document]:
        """
        Load a single file and split into chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document chunks
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Select loader based on file extension
        extension = path.suffix.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(str(path))
        elif extension == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Load and split
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Add source metadata
        for chunk in chunks:
            chunk.metadata["source"] = path.name
            chunk.metadata["file_path"] = str(path)
        
        print(f"✅ Loaded {path.name}: {len(chunks)} chunks")
        return chunks
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*.*") -> list[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern to match files
            
        Returns:
            List of Document chunks
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_chunks = []
        supported_extensions = {".pdf", ".txt", ".docx", ".doc"}
        
        for file_path in path.glob(glob_pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.load_file(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"⚠️ Error loading {file_path.name}: {e}")
        
        print(f"✅ Loaded {len(all_chunks)} total chunks from {directory_path}")
        return all_chunks
    
    def load_text(self, text: str, metadata: dict = None) -> list[Document]:
        """
        Load raw text and split into chunks.
        
        Args:
            text: Raw text content
            metadata: Optional metadata dict
            
        Returns:
            List of Document chunks
        """
        chunks = self.text_splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata=metadata or {})
            for chunk in chunks
        ]
        return documents


if __name__ == "__main__":
    print("=" * 50)
    print("DOCUMENT LOADER TEST")
    print("=" * 50)
    
    loader = DocumentLoader()
    
    # Test with raw text
    test_text = """
    LangChain is a framework for developing applications powered by language models.
    It provides tools for building agents, chains, and RAG systems.
    
    Key features:
    - Agents: LLMs that can use tools
    - Chains: Sequences of operations
    - Memory: Remember conversation history
    - RAG: Combine retrieval with generation
    
    LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.
    """
    
    chunks = loader.load_text(test_text, metadata={"source": "test"})
    
    print(f"\n📄 Created {len(chunks)} chunks from test text:")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. {chunk.page_content[:60]}...")
    
    print("\n✅ Test completed!")
