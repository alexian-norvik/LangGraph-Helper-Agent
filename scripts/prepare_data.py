#!/usr/bin/env python3
"""One-time data preparation script.

Downloads documentation files, preprocesses them, and creates the vector store.

Usage:
    python scripts/prepare_data.py           # Download, preprocess, and index docs
    python scripts/prepare_data.py --force   # Force re-download and re-index
    python scripts/prepare_data.py --no-preprocess  # Skip preprocessing
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.common.constants import HEADER_WIDTH  # noqa: E402
from src.config import VECTORSTORE_DIR  # noqa: E402
from src.data_prep.chunker import chunk_documents  # noqa: E402
from src.data_prep.downloader import download_docs, get_all_docs  # noqa: E402
from src.data_prep.preprocessor import preprocess_all_docs  # noqa: E402
from src.data_prep.vectorstore import create_vectorstore, vectorstore_exists  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LangGraph Helper Agent")
    parser.add_argument(
        "--force", action="store_true", help="Force re-download and re-index even if data exists"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Only download docs, don't create vector store"
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", help="Skip preprocessing (use raw docs)"
    )
    args = parser.parse_args()

    logger.info("=" * HEADER_WIDTH)
    logger.info("LangGraph Helper Agent - Data Preparation")
    logger.info("=" * HEADER_WIDTH)

    # Step 1: Download documentation
    logger.info("\n[1/3] Downloading documentation files...")
    results = download_docs(force=args.force)

    if not any(results.values()):
        logger.error("Failed to download any documentation files")
        sys.exit(1)

    successful = sum(1 for v in results.values() if v)
    logger.info(f"\nDownload complete: {successful}/{len(results)} files")

    if args.download_only:
        logger.info("\n--download-only specified, skipping vector store creation")
        return

    # Step 2: Load and preprocess docs
    logger.info("\n[2/3] Loading and preprocessing documentation...")
    docs = get_all_docs()
    if not docs:
        logger.error("No documentation files found")
        sys.exit(1)

    if args.no_preprocess:
        logger.info("  Skipping preprocessing (--no-preprocess)")
    else:
        logger.info("  Preprocessing: removing JS/TS, cleaning markdown, filtering...")
        docs = preprocess_all_docs(docs)

    total_chars = sum(len(c) for c in docs.values())
    logger.info(f"  Total content: {total_chars:,} characters across {len(docs)} files")

    # Step 3: Create vector store
    logger.info("\n[3/3] Creating vector store...")

    if vectorstore_exists() and not args.force:
        logger.info(f"  Vector store already exists at {VECTORSTORE_DIR}")
        logger.info("  Use --force to rebuild")
        return

    chunks = chunk_documents(docs)
    logger.info(f"  Created {len(chunks)} chunks")

    create_vectorstore(chunks, VECTORSTORE_DIR)

    logger.info("\n" + "=" * HEADER_WIDTH)
    logger.info("Data preparation complete!")
    logger.info("=" * HEADER_WIDTH)
    logger.info("\nYou can now run the agent:")
    logger.info("  python main.py                    # Interactive mode")
    logger.info("  python main.py 'Your question'    # Quick query")


if __name__ == "__main__":
    main()
