"""Text chunking for documentation files."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def create_splitter(
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    """Create a text splitter configured for markdown documentation.

    Args:
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        Configured text splitter
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n## ",  # H2 headers
            "\n### ",  # H3 headers
            "\n#### ",  # H4 headers
            "\n\n",  # Paragraphs
            "\n",  # Lines
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters
        ],
        keep_separator=True,
    )


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_size: int = 100,
) -> list[Document]:
    """Split text into chunks with metadata.

    Args:
        text: The text content to chunk
        source: Source identifier for metadata
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size (filters out header-only chunks)

    Returns:
        List of Document objects with chunked content
    """
    splitter = create_splitter(chunk_size, chunk_overlap)

    # Create a single document and split it
    doc = Document(page_content=text, metadata={"source": source})
    chunks = splitter.split_documents([doc])

    # Filter out tiny chunks (usually just headers)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chunk_size]

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def chunk_documents(docs: dict[str, str]) -> list[Document]:
    """Chunk multiple documents.

    Args:
        docs: Dictionary mapping source names to document content

    Returns:
        List of all Document chunks from all sources
    """
    all_chunks = []

    for source, content in docs.items():
        logger.info(f"Chunking {source}...")
        chunks = chunk_text(content, source)
        logger.info(f"Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    from src.data_prep.downloader import get_all_docs

    docs = get_all_docs()
    if docs:
        chunks = chunk_documents(docs)
        logger.info(f"Sample chunk:\n{chunks[0].page_content[:500]}...")
    else:
        logger.warning("No docs found. Run downloader first.")
