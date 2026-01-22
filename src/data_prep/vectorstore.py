"""FAISS vector store management."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY, VECTORSTORE_DIR


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get the embedding model.

    Returns:
        Configured embedding model
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def create_vectorstore(documents: list[Document], save_path: Path | None = None) -> FAISS:
    """Create a FAISS vector store from documents.

    Args:
        documents: List of Document objects to index
        save_path: Optional path to save the vector store

    Returns:
        FAISS vector store
    """
    if not documents:
        raise ValueError("No documents provided to create vector store")

    print(f"Creating vector store with {len(documents)} documents...")
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        print(f"Vector store saved to {save_path}")

    return vectorstore


def load_vectorstore(load_path: Path | None = None) -> FAISS | None:
    """Load a FAISS vector store from disk.

    Args:
        load_path: Path to load from (defaults to VECTORSTORE_DIR)

    Returns:
        FAISS vector store, or None if not found
    """
    load_path = Path(load_path) if load_path else VECTORSTORE_DIR

    index_file = load_path / "index.faiss"
    if not index_file.exists():
        print(f"No vector store found at {load_path}")
        return None

    print(f"Loading vector store from {load_path}...")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully")
    return vectorstore


def vectorstore_exists(path: Path | None = None) -> bool:
    """Check if a vector store exists at the given path.

    Args:
        path: Path to check (defaults to VECTORSTORE_DIR)

    Returns:
        True if vector store exists
    """
    path = Path(path) if path else VECTORSTORE_DIR
    return (path / "index.faiss").exists()


if __name__ == "__main__":
    from src.data_prep.chunker import chunk_documents
    from src.data_prep.downloader import get_all_docs

    docs = get_all_docs()
    if docs:
        chunks = chunk_documents(docs)
        vs = create_vectorstore(chunks, VECTORSTORE_DIR)

        # Test search
        results = vs.similarity_search("How do I create a StateGraph?", k=3)
        print("\nTest search results:")
        for i, doc in enumerate(results):
            print(f"\n{i+1}. [{doc.metadata['source']}]")
            print(f"   {doc.page_content[:200]}...")
    else:
        print("No docs found. Run downloader first.")
