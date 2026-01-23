"""Download llms.txt documentation files."""

from pathlib import Path

import requests
from loguru import logger

from src.common.constants import DOWNLOAD_TIMEOUT
from src.config import DATA_DIR, DOC_FILES, DOC_URLS


def download_file(url: str, filepath: Path) -> bool:
    """Download a file from URL to the specified filepath.

    Args:
        url: The URL to download from
        filepath: The local path to save the file

    Returns:
        True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url}...")
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(response.text, encoding="utf-8")

        logger.info(f"Saved to {filepath} ({len(response.text):,} bytes)")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_docs(force: bool = False) -> dict[str, bool]:
    """Download all documentation files.

    Args:
        force: If True, re-download even if files exist

    Returns:
        Dictionary mapping doc names to download success status
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, url in DOC_URLS.items():
        filepath = DOC_FILES[name]

        if filepath.exists() and not force:
            logger.info(f"Skipping {name}: {filepath} already exists (use --force to re-download)")
            results[name] = True
            continue

        results[name] = download_file(url, filepath)

    return results


def get_all_docs() -> dict[str, str]:
    """Get content of all downloaded documentation files.

    Returns:
        Dictionary mapping doc names to their content
    """
    docs = {}
    for name, filepath in DOC_FILES.items():
        if filepath.exists():
            docs[name] = filepath.read_text(encoding="utf-8")
    return docs


if __name__ == "__main__":
    download_docs(force=True)
