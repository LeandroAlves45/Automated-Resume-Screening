# ==========================================================
# resume_parser.py - CV file Parser
# ==========================================================
# Responsability: Extract plain text from CV files in various formats (PDF, DOCX, TXT).
#
# Design pattern applied: Facade Pattern
# This class hides the complexity of three different parsing libraries behind a single, uniform interface.
# The caller never needs to known which library is being used - it always gets back the same dict structure.
#
# SOLID Principle: Single Responsibility Principle (SRP)
# Each private method handles exactly one file format. The public methods handle routing and folder traversal. 
# ==========================================================

import logging
from pathlib import Path 
from pdfminer.high_level import extract_text as pdf_extract_text # PDF parsing
from docx import Document # DOCX parsing

from config import SUPPORTED_EXTENSIONS, TEXT_CONFIG

# Configure logging for this module
logger = logging.getLogger(__name__)

class ResumeParser:
    """
    Parses CV files (PDF, DOCX, TXT) and returns extracted plain text for further processing.

    Supports single file parsing via parse() and batch folder parsing via parse_folder().
    Files that fail to parse are reported without aborting the entire process.(NFR-06: Robustness)
    """

    #-----------------------------
    # PUBLIC INTERFACE
    #-----------------------------

    def parse(self, file_path: str) -> dict:
        """
        Parse a single CV file and return its extracted text.
        Determines the file format by extension and delegates to the appropriate private parsing method.
        If parsing fails for any reason, the error is captured and returned  in the dict rather thanraised- this prevents one bad file from aborting a batch.
        Args:
            file_path (str): Absolute or relative path to the CV file.
            
        Returns:
            dict: {
                "name": str,  # The file name (without path)
                "file": str,  # The full file path as string
                "text": str,  # The extracted plain text from the CV (empty on failure)
                "error": str  # An error message if parsing failed, otherwise None
            }
        """

        # Convert to a Path object for reliable cross-platform path handling
        path = Path(file_path)

        # Build the base result dict.
        result = {
            "name": path.stem, # File name without extension
            "file": str(path), # Full file path as string
            "text": "", # Will be filled with extracted text or remain empty on failure
            "error": None # Will be filled with an error message if parsing fails
        } 

        # --- Existence check ---
        # Early check to see if the file exists before attempting to parse. If it doesn't, we can skip all parsing logic and return an error immediately.
        if not path.exists():
            result["error"] = f"File not found: {file_path}"
            logger.warning(result["error"])
            return result
        
        # --- Unsupported extension check ---
        # Only attempt to parse files with supported extensions. If the extension is not supported, we can skip parsing and return an error immediately.
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            result["error"] = f"Unsupported file type: {extension}"
            logger.warning(result["error"])
            return result
        
        # --- Dispatch to the correct parser based on file extension ---
        # Each parsing method handles a specific format. Cath here any exceptions they might raise and capture them in the result dict to ensure robustness.
        try: 
            if extension == ".pdf":
                raw_text = self._parse_pdf (str(path))
            elif extension == ".docx":
                raw_text = self._parse_docx(str(path))
            elif extension == ".txt":
                raw_text = self._parse_txt(str(path))
            
            #Clean the extracted text before storing it
            # Raw text from parsers often contains excessives whitespace, duplicate blank lines, and another noise
            result["text"] = self._clean_text(raw_text)

            # --- Minimum lenght validation (Business Rule) ---
            # If the CV has fewer characters than the minimum (after cleaning), is considered invalid. 
            # This catches scanned PDFs (image-only files) that produce empty or near-empty text after extraction.
            min_length = TEXT_CONFIG["min_resume_length"]  
            if len(result["text"]) < min_length:
                result["error"] = (
                    f"Extracted text too short: ({len(result['text'])} chars."
                    f"Minimum required is {min_length}. File may be image-only or empty. "
                )
                result["text"] = "" # Clear the text to avoid processing invalid data
                logger.warning(f"{path.name}: {result['error']}")
            else:
                logger.info(f"Successfully parsed: {path.name}"
                            f"({len(result['text'])} chars.)")
        except Exception as e:
            # Cath-all: any unexpected library error is recorded here.
            # We log the full exception for debugging.
            result["error"] = f"Parsing failed:{type(e).__name__}: {e}"
            logger.error(f"{path.name}: {result['error']}")

        return result
    
    def parse_folder(self, folder_path:str) -> list[dict]:
        """
        Parse all supported CV found in a folder
        
        Iterates over all files in the folder, attempts to parse each one,and collects the results.
        Unsupported files are skipped silently. Files that fail are included in the results with their error field. 
        They are never dropped and receive why they failed.
        """

        folder = Path(folder_path)

        # Validate if the folder exists before iterate it.
        if not folder.is_dir():
            logger.error(f"Folder not found or is not a directory: {folder_path}")
            return []
        
        results = []

        # Iterate all files in the folder (non-recursive - subfolders are ignored)
        # Sort the files to ensure consistent ordering across operating systems since directory listing order are not guaranteed.
        for file_path in sorted(folder.iterdir()):

            # Skip subfolders - only process files
            if not file_path.is_file():
                continue

            # Skip files that are not supported without logging or warning - this would be noisy if the folder contains many non-CV files
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logging.debug(f"Skipping unsupported file: {file_path.name}")
                continue

            # parse() handles all error cases internally and always returns a dict.
            # Never need a try/except. parse() handles is robustness.
            result = self.parse(str(file_path))
            results.append(result)

        logger.info(f"Folder scan complete: {len(results)} supported file(s) found."
                    f"In {folder_path}")
        return results
    
    #-----------------------------
    # PRIVATE METHODS - one per file
    #-----------------------------

    def _parse_pdf(self, path: str) -> str:
        """
        Extract text from a PDF file using pdfminer.six.
        
        pdfminer.six is a pure-Python library that analyses the internal 
        structure of a PDF to extract text. It works well for digitally created
        PDFs but produces empty or garbled output for scanned PDFs (image-only files)
        
        Args:
            path (str): The file path to the PDF file.
            
        Returns:
            str: Raw extracted text. May contain excessive whitespace - caller
            is responsible for passing this through _clean_text().
        """

        # extract_text is pdfminer's high-level API - it handles page iteration
        # and character decoding internally. We use directly because the low-level API
        # would require managing objects manually with no benefit.
        text = pdf_extract_text(path)

        # pdfminer returns None if the PDF has no extractable text layer
        return text if text is not None else ""

    def _parse_docx(self, path:str) -> str:
        """
        Extract text from a DOCX file using python-docx.

        python-docx reads the XML structure of a .docx file and exposes its content
        as paragraph objects. We join all paragraphs with newlines to preserve the logical structure of the document.

        Known limitation: Text inside tables, headers, footers, and text boxes is not captured by this approach.
        For v1.0, this is acceptable as most CVs use paragraphs for their main content.

        Args:
            path (str): Path to the DOCX file.

        Returns: 
            str: Raw extracted text with paragraphs separated newlines.
        """

        doc = Document(path)

        # Each paragraph in the DOCX is a separate object with a .text attribute
        # We join them with '\n' rather than '' to preserve the visual separation btween sections
        # Empty paragraphs (blank line in the document) are included as empty
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs)

    def _parse_txt(self, path: str) -> str:
        """
        Read a plain text CV file with encoding fallback.

        Many CVs saved as .txt use UTF-8 encodng, but older files or files created on Windows
        may use Latin-1 (ISO-8859-1). We try UTF-8 first (the correct standard),  and fall back to Latin-1
        if UTF-( decoding fails).
        
        Args:
            path (str): Path to the TXT file.

        Returns:
            str: File contents as string
        """

        # First attempt: UTF-8, which handles most modern files correctly.
        # errors='strict' means a UnicodeDecodeError is raised on any invalid byte.
        # which we intentionally catch to trigger the Latin-1 fallback.
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                return f.read()

        except UnicodeDecodeError:
            # Second attempt: Latin-1
            logger.warning(f"UTF-8 decoding failed for '{path}'."
                           f"Retrying with Latin-1 encoding.")
            with open(path, "r", encoding="latin-1") as f:
                return f.read()

    #-----------------------------
    # TEXT CLEANING
    #-----------------------------

    def _clean_text(self, text: str) -> str:
        """
        Normalise raw extracted text by removing excessive whitespace and blank lines.

        After extraction, text often contains long runs of spaces (from PDF column alingnment),
        tabs, and many consecutive blank lines. This method collapses them into a clean, readable,
        format that is easier for the NLP pipeline to process downstream.

        Args:
            text (str): Raw text as returned by a parse method

        Returns:
            str: Cleaned text with normalised whitespace.
        """

        if not text:
            return ""

        #split into individual lines to process them one by one
        lines = text.splitlines()

        cleaned_lines = []
        for line in lines:
            # Replace any sequence of whitespace characters (tabs, multiple spaces)
            # within a line with a single space. This fixes PDF extraction artefacts
            # where columns are separated by many spaces instead of a tab or newline
            cleaned_line = " ".join(line.split())
            cleaned_lines.append(cleaned_line)

        # Rejoin lines and collapse sequences of more than 2 consecutive blank lines
        # into a single blank line. This preserves section separation (one blank line between secions is meaningful)
        # without leaving large empty gaps
        result = "\n".join(cleaned_lines)

        # Replace 3 or more consecutive newlines with exactly 2.
        import re
        result = re.sub(r"\n{3,}", "\n\n", result)

        # Strip leading and trailing whitespace from the entire document
        return result.strip()

        