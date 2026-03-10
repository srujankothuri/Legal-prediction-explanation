"""
PDF text extraction for Indian legal documents.

Extracts full text, act sections, and headnotes from
Supreme Court judgment PDFs (typically from Indian Kanoon).

Usage:
    from src.data.pdf_extractor import PDFExtractor

    extractor = PDFExtractor("path/to/judgment.pdf")
    print(extractor.full_text)
    print(extractor.headnote)
    print(extractor.act_section)
"""

import re
from typing import Optional
from PyPDF2 import PdfReader

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extract structured text from Indian legal judgment PDFs.

    Parses out:
    - Full text content
    - Act section (text after "act:" marker)
    - Headnote (text between "headnote:" and "judgment:" markers)
    """

    # Pattern to remove Indian Kanoon watermarks
    WATERMARK_PATTERN = re.compile(
        r"indian kanoon - http://indiankanoon\.org/.*?\n",
        re.IGNORECASE
    )

    def __init__(self, pdf_path: str):
        """
        Extract text from PDF on initialization.

        Args:
            pdf_path: Path to the PDF file

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF has no extractable text
        """
        self.pdf_path = pdf_path
        logger.info(f"Extracting text from: {pdf_path}")

        self.full_text = self._extract_full_text()
        self.act_section = self._extract_act_section()
        self.headnote = self._extract_headnote()

        word_count = len(self.full_text.split())
        logger.info(f"Extracted {word_count} words from PDF")

    def _extract_full_text(self) -> str:
        """Extract all text from all pages of the PDF."""
        try:
            reader = PdfReader(self.pdf_path)
        except Exception as e:
            logger.error(f"Failed to read PDF: {e}")
            raise ValueError(f"Could not read PDF: {self.pdf_path}") from e

        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i}: {e}")

        if not pages_text:
            raise ValueError(f"No extractable text found in PDF: {self.pdf_path}")

        full_text = "\n".join(pages_text)

        # Clean watermarks
        full_text = self.WATERMARK_PATTERN.sub("", full_text)

        return full_text

    def _extract_act_section(self) -> Optional[str]:
        """Extract the Act section from the document."""
        text_lower = self.full_text.lower()
        match = re.search(r"act:(.*)", text_lower, re.DOTALL | re.IGNORECASE)

        if match:
            act_text = match.group(1).strip()
            act_text = self.WATERMARK_PATTERN.sub("", act_text)
            logger.debug(f"Act section found: {len(act_text)} chars")
            return act_text

        logger.debug("Act section not found in document")
        return None

    def _extract_headnote(self) -> Optional[str]:
        """Extract the headnote (between 'headnote:' and 'judgment:' markers)."""
        text_lower = self.full_text.lower()
        match = re.search(
            r"headnote:(.*?)judgment:",
            text_lower,
            re.DOTALL | re.IGNORECASE
        )

        if match:
            headnote = match.group(1).strip()
            headnote = self.WATERMARK_PATTERN.sub("", headnote)
            logger.debug(f"Headnote found: {len(headnote)} chars")
            return headnote

        logger.debug("Headnote not found in document")
        return None

    def get_text_for_prediction(self) -> str:
        """
        Get the best available text for model prediction.
        Prefers act_section (contains the judgment body),
        falls back to full_text.

        Returns:
            Text string suitable for model input
        """
        if self.act_section and len(self.act_section) > 100:
            return self.act_section
        return self.full_text

    def summary(self) -> str:
        """Generate a summary of extracted content."""
        lines = [
            f"PDF: {self.pdf_path}",
            f"Full text: {len(self.full_text)} chars, {len(self.full_text.split())} words",
            f"Act section: {'Found' if self.act_section else 'Not found'}",
            f"Headnote: {'Found' if self.headnote else 'Not found'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        word_count = len(self.full_text.split())
        return f"PDFExtractor({self.pdf_path}, {word_count} words)"