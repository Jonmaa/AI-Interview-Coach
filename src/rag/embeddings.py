"""
Embeddings module using HuggingFace sentence-transformers (100% free, local).
Creates vector representations of text for semantic search without any paid API.

Uses sentence-transformers/all-MiniLM-L6-v2 by default:
- 384 dimensions, fast inference, excellent quality
- Runs entirely on CPU/GPU locally
- No API keys or costs required
"""

from langchain_huggingface import HuggingFaceEmbeddings


# Available free embedding models (all run locally):
EMBEDDING_MODELS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",       # 384d, ~80MB, fastest
    "balanced": "sentence-transformers/all-mpnet-base-v2",   # 768d, ~420MB, best quality/speed
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384d, multilingual
}


def get_embeddings(
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """
    Get a free, local HuggingFace embeddings model.

    Args:
        model: HuggingFace model name or a key from EMBEDDING_MODELS
               (e.g. "fast", "balanced", "multilingual").
        device: "cpu" or "cuda" for GPU acceleration.

    Returns:
        HuggingFaceEmbeddings instance (drop-in replacement for OpenAIEmbeddings)
    """
    # Allow friendly aliases
    model_name = EMBEDDING_MODELS.get(model, model)

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    print("=" * 50)
    print("EMBEDDINGS TEST (Free - HuggingFace)")
    print("=" * 50)

    embeddings = get_embeddings()

    # Test embedding a text
    test_text = "What is LangChain and how do you use it for building agents?"
    vector = embeddings.embed_query(test_text)

    print(f"✅ Embedding created successfully!")
    print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   Text: '{test_text[:50]}...'")
    print(f"   Vector dimensions: {len(vector)}")
    print(f"   First 5 values: {vector[:5]}")
    print(f"   💰 Cost: $0.00 (runs locally)")
