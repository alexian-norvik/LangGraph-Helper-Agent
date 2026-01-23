"""Preprocessing pipeline for documentation files.

Cleans and filters documentation before chunking to improve retrieval quality.
"""

import re

from loguru import logger

from src.common.constants import (
    DEDUP_SIGNATURE_LENGTH,
    EMPTY_CODE_BLOCK_PATTERN,
    MAX_CONSECUTIVE_BLANK_LINES,
    MIN_SECTION_LENGTH,
    NAVIGATION_REMOVAL_PATTERNS,
    PYTHON_CODE_INDICATORS,
    WHITESPACE_CODE_BLOCK_PATTERN,
)


def normalize_code_blocks(text: str) -> str:
    """Normalize code block markers for consistency.

    Keeps both Python and TypeScript/JavaScript code blocks.
    Only removes clearly broken or empty code blocks.

    Args:
        text: Raw documentation text

    Returns:
        Text with normalized code blocks
    """
    # Remove empty code blocks
    text = re.sub(EMPTY_CODE_BLOCK_PATTERN, "", text)

    # Remove code blocks that are just whitespace
    text = re.sub(WHITESPACE_CODE_BLOCK_PATTERN, "", text)

    return text


def remove_navigation_boilerplate(text: str) -> str:
    """Remove navigation links, breadcrumbs, and boilerplate.

    Args:
        text: Documentation text

    Returns:
        Cleaned text
    """
    for pattern in NAVIGATION_REMOVAL_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    return text


def clean_markdown(text: str) -> str:
    """Clean up markdown formatting issues.

    Args:
        text: Documentation text

    Returns:
        Cleaned text
    """
    # Remove excessive blank lines (more than allowed)
    text = re.sub(rf"\n{{{MAX_CONSECUTIVE_BLANK_LINES},}}", "\n\n\n", text)

    # Remove trailing whitespace on lines
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Clean up broken markdown links
    text = re.sub(r"\[([^\]]+)\]\(\s*\)", r"\1", text)

    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove inline HTML tags (but keep content)
    text = re.sub(r"<(?:div|span|p|br|hr)[^>]*/?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:div|span|p)>", "", text, flags=re.IGNORECASE)

    return text


def extract_python_sections(text: str) -> str:
    """Enhance Python code sections with clear markers.

    Args:
        text: Documentation text

    Returns:
        Text with enhanced Python sections
    """

    # Ensure Python code blocks are clearly marked
    # Convert generic code blocks that look like Python
    def convert_generic_to_python(match):
        code = match.group(1)
        # Check if it looks like Python
        if any(indicator in code for indicator in PYTHON_CODE_INDICATORS):
            return f"```python\n{code}```"
        return match.group(0)

    text = re.sub(r"```\n(.*?)```", convert_generic_to_python, text, flags=re.DOTALL)

    return text


def remove_duplicate_sections(text: str) -> str:
    """Remove duplicate content sections.

    Args:
        text: Documentation text

    Returns:
        Text with duplicates removed
    """
    # Split into sections by headers
    sections = re.split(r"(^#{1,4}\s+.+$)", text, flags=re.MULTILINE)

    seen_content = set()
    unique_sections = []

    for section in sections:
        # Skip very short sections
        if len(section.strip()) < MIN_SECTION_LENGTH:
            unique_sections.append(section)
            continue

        signature = section.strip()[:DEDUP_SIGNATURE_LENGTH].lower()
        if signature not in seen_content:
            seen_content.add(signature)
            unique_sections.append(section)

    return "".join(unique_sections)


def filter_relevant_content(text: str) -> str:
    """Filter out irrelevant content sections.

    Args:
        text: Documentation text

    Returns:
        Filtered text
    """
    # Remove sections that are clearly not useful
    remove_patterns = [
        # API reference with just type signatures (no examples)
        r"#{2,4}\s+(?:Parameters|Returns|Raises|Attributes)\s*\n(?:\s*[-*]\s+\*\*\w+\*\*.*\n)+",
        # Long lists of just links
        r"(?:^\s*[-*]\s+\[[\w\s]+\]\([^)]+\)\s*$\n){5,}",
        # Version/changelog sections
        r"#{2,4}\s+(?:Changelog|Version History|Release Notes).*?(?=^#{1,3}\s|\Z)",
        # Installation for other languages
        r"#{2,4}\s+(?:npm|yarn|pnpm)\s+install.*?(?=^#{1,3}\s|\Z)",
    ]

    for pattern in remove_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)

    return text


def preprocess_document(text: str, source: str) -> str:
    """Run full preprocessing pipeline on a document.

    Args:
        text: Raw documentation text
        source: Source identifier

    Returns:
        Preprocessed text ready for chunking
    """
    original_len = len(text)

    # Step 1: Normalize code blocks (keep both Python and TypeScript)
    text = normalize_code_blocks(text)
    logger.debug(
        f"After code normalization: {len(text)} chars ({len(text) / original_len * 100:.1f}%)"
    )

    # Step 2: Remove navigation and boilerplate
    text = remove_navigation_boilerplate(text)

    # Step 3: Clean markdown formatting
    text = clean_markdown(text)

    # Step 4: Enhance Python sections
    text = extract_python_sections(text)

    # Step 5: Filter irrelevant content
    text = filter_relevant_content(text)

    # Step 6: Remove duplicates
    text = remove_duplicate_sections(text)

    final_len = len(text)
    reduction = (1 - final_len / original_len) * 100
    logger.info(
        f"Preprocessed {source}: {original_len:,} -> {final_len:,} chars ({reduction:.1f}% reduction)"
    )

    return text.strip()


def preprocess_all_docs(docs: dict[str, str]) -> dict[str, str]:
    """Preprocess all documents.

    Args:
        docs: Dictionary mapping source names to document content

    Returns:
        Dictionary with preprocessed content
    """
    preprocessed = {}

    for source, content in docs.items():
        logger.info(f"Preprocessing {source}...")
        preprocessed[source] = preprocess_document(content, source)

    return preprocessed


if __name__ == "__main__":
    from src.data_prep.downloader import get_all_docs

    docs = get_all_docs()
    if docs:
        preprocessed = preprocess_all_docs(docs)

        # Show sample
        for source, content in preprocessed.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Source: {source}")
            logger.info(f"Length: {len(content):,} chars")
            logger.info(f"{'=' * 60}")
            logger.info(content[:1000])
            logger.info("...")
    else:
        logger.warning("No docs found. Run downloader first.")
