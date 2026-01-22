"""FAISS vector store management."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR


def get_embeddings() -> OllamaEmbeddings:
    """Get the Ollama embedding model.

    Returns:
        Configured Ollama embedding model
    """
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def create_vectorstore(
    documents: list[Document],
    save_path: Path | None = None,
    batch_size: int = 100,
) -> FAISS:
    """Create a FAISS vector store from documents with progress logging.

    Args:
        documents: List of Document objects to index
        save_path: Optional path to save the vector store
        batch_size: Number of documents to process per batch

    Returns:
        FAISS vector store
    """
    if not documents:
        raise ValueError("No documents provided to create vector store")

    total_docs = len(documents)
    logger.info(f"Creating vector store with {total_docs} documents (batch size: {batch_size})...")
    embeddings = get_embeddings()

    # Process first batch to initialize vectorstore
    first_batch = documents[:batch_size]
    logger.info(
        f"Processing batch 1/{(total_docs + batch_size - 1) // batch_size} (documents 1-{len(first_batch)})..."
    )
    vectorstore = FAISS.from_documents(first_batch, embeddings)
    logger.info(
        f"Progress: {len(first_batch)}/{total_docs} documents embedded ({len(first_batch) * 100 // total_docs}%)"
    )

    # Process remaining batches
    for i in range(batch_size, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size
        logger.info(
            f"Processing batch {batch_num}/{total_batches} (documents {i + 1}-{i + len(batch)})..."
        )
        vectorstore.add_documents(batch)
        processed = i + len(batch)
        logger.info(
            f"Progress: {processed}/{total_docs} documents embedded ({processed * 100 // total_docs}%)"
        )

    logger.info("Embedding complete!")

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        logger.info(f"Vector store saved to {save_path}")

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
        logger.warning(f"No vector store found at {load_path}")
        return None

    logger.info(f"Loading vector store from {load_path}...")
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(str(load_path), embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector store loaded successfully")
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
        logger.info("Test search results:")
        for i, doc in enumerate(results):
            logger.info(f"{i+1}. [{doc.metadata['source']}]")
            logger.info(f"   {doc.page_content[:200]}...")
    else:
        logger.warning("No docs found. Run downloader first.")
