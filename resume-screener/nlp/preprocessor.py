# =============================================================================
# preprocessor.py — Text Preprocessor
# =============================================================================
# Responsibility: Clean, normalise, and tokenise raw CV text so that it is
# ready for feature extraction by extractor.py.
#
# This module sits between the parsers (which extract raw text) and the
# extractor (which extracts meaningful features). Its job is purely
# transformation — no interpretation, no scoring.
#
# Design principle: Single Responsibility Principle (SRP).
# This module does one thing: transform raw text into clean, normalised
# tokens. Feature extraction and scoring are strictly other modules' concern.
#
# Design pattern: Dependency Injection (same as jd_parser.py).
# The spaCy model is passed in via the constructor, not loaded internally.
# This ensures the model is only loaded once across the entire pipeline.
# =============================================================================

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Cleans and normalises raw text extracted from CV files.
 
    Produces two representations of the input text:
      1. A clean plain-text string (for regex-based feature extraction)
      2. A list of lemmatised tokens (for TF-IDF vectorisation)
 
    Both representations are derived from the same input text and returned
    together via the process() method, so callers never need to call
    multiple methods to get a fully prepared document.
    """

    def __init__(self, nlp: Any) -> None:
        """
        Initialise the preprocessor with a pre-loaded spaCy language model.
 
        Args:
            nlp: A loaded spaCy Language object, injected from main.py.
                 Must be the same instance shared across all NLP modules.
        """
        self._nlp = nlp
        logger.debug("TextPreprocessor initialised.")

    # -------------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------------

    def process(self, raw_text: str) -> dict:
        """
        Run the full preprocessing pipeline on raw CV text.
 
        Combines two sub-processes:
          1. _clean_text()   → produces a normalised plain-text string
          2. _tokenise()     → produces a list of lemmatised keyword tokens
 
        The clean text is used by the extractor for regex-based operations
        (finding years of experience, education level, etc.).
        The tokens are used by the scorer's TF-IDF vectoriser.
 
        Args:
            raw_text (str): Raw text as returned by ResumeParser.parse().
 
        Returns:
            dict: {
                "clean_text": str,       # Normalised plain text string
                "tokens":     list[str]  # Lemmatised, filtered keyword tokens
            }
        """
        # Guard: if the input is empty, return safe empty defaults immediately.
        # There is no point running the spaCy pipeline on empty text — it would
        # produce an empty doc and waste compute time.
        if not raw_text or not raw_text.strip():
            logger.warning("Empty text passed to preprocessor. Returning empty results.")
            return {"clean_text": "", "tokens": []}
        
        # Step 1: Clean and normalise the raw text.
        # This produces a human-readable, normalised version of the CV text.
        clean_text = self._clean_text(raw_text)

        # Step 2: Run the spaCy pipeline on the clean text to get tokens.
        # We run spaCy on the CLEAN text (not raw) so that noise like
        # excessive punctuation and encoding artefacts don't confuse the
        # tokeniser and lemmatiser.
        doc = self._nlp(clean_text)
        tokens = self._tokenise(doc)

        logger.debug(
            f"Precprocessing complete -"
            f"Clean text: {len(clean_text)} chars, Tokens: {len(tokens)}"
        )

        return {
            "clean_text": clean_text,
            "tokens": tokens
        }
    
    # -------------------------------------------------------------------------------
    # PRIVATE METHODS
    # -------------------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """
        Normalise raw text by removing noise and standardising formatting.
 
        Applies a sequence of regex substitutions and string operations to
        produce text that is easier to work with for both regex-based
        extraction (in extractor.py) and NLP (via spaCy).
 
        The order of operations matters:
          1. Remove non-printable characters first (before other substitutions
             could accidentally match them)
          2. Normalise punctuation and symbols
          3. Collapse whitespace last (so previous steps don't introduce
             new whitespace issues that need to be cleaned again)
 
        Args:
            text (str): Raw text from the resume parser.
 
        Returns:
            str: Cleaned, normalised text.
        """
        # --- Step 1: Remove non-printable and control characters ---
        # These are characters with ASCII values below 32 (space), except
        # for \n (newline, ASCII 10) and \t (tab, ASCII 9) which we want
        # to keep as structural separators.
        # \x00-\x08 and \x0b-\x1f covers all control characters except \t and \n.
        text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)

        # --- Step 2: Normalise common Unicode punctuation to ASCII equivalents ---
        # CVs created in Word or copied from websites often contain "smart quotes"
        # (\u2018, \u2019, \u201c, \u201d), em-dashes (\u2014), and other Unicode
        # punctuation. We normalise these to standard ASCII so regex patterns
        # written with simple characters work correctly downstream.
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # Smart single quotes → straight single quote

        text = text.replace("\u201c", '"').replace("\u201d", '"')  # Smart double quotes → straight double quote

        text = text.replace("\u2013", "-").replace("\u2014", "-")  # En dash and em dash → hyphen

        text = text.replace("\u2022", "-").replace("\u00b7", "-")  # Bullet points → hyphen

        # --- Step 3: Normalise email addresses and URLs ---
        # Emails and URLs contribute no meaningful semantic signal for our
        # scoring criteria (they don't tell us about skills or experience),
        # but they do introduce noise tokens that inflate the keyword list.
        # We replace them with a placeholder rather than deleting them entirely,
        # so that surrounding context (spaces, sentence boundaries) is preserved.
        text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)  # Emails
        text = re.sub(r"https?://\S+", " URL ", text)  # HTTP/HTTPS URLs
        text = re.sub(r"www\.\S+", " URL ", text)  # URLs without scheme

        # --- Step 4: Collapse multiple spaces into a single space ---
        # Previous substitutions may have introduced extra spaces. This step
        # normalises them before we process newlines.
        text = re.sub(r"[ \t]+", " ", text)

        # --- Step 5: Normalise newlines ---
        # Collapse runs of 3 or more newlines into exactly 2 (one blank line).
        # We keep up to 2 newlines because a blank line is semantically useful
        # — it separates sections (experience from education, etc.).
        text = re.sub(r"\n{3,}", "\n\n", text)

        # --- Step 6: Final strip ---
        # Remove any leading or trailing whitespace from the entire document.
        return text.strip()

    def _tokenise(self, doc: Any) -> list[str]:
        """
        Convert a spaCy Doc into a list of lemmatised, filtered keyword tokens.
 
        Filters out tokens that carry no meaningful semantic signal:
          - Stop words (the, a, is, of, ...)
          - Punctuation and whitespace tokens
          - Numeric-only tokens (e.g. phone numbers, zip codes)
          - Tokens shorter than 2 characters
 
        Then lemmatises the remaining tokens (reduces to base form) and
        lowercases them, so that "developed", "developing", and "developer"
        all contribute to the same term in the TF-IDF vocabulary.
 
        Args:
            doc: A spaCy Doc object produced by running nlp(clean_text).
 
        Returns:
            list[str]: Filtered, lemmatised, lowercased tokens.
                       Duplicates are retained — TF-IDF needs term frequency.
        """
        tokens = []

        for token in doc:
            # --- Filter 1: Stop words ---
            # spaCy's stop word list for English includes ~326 common words.
            # These words appear in virtually every document and carry no
            # discriminating signal for comparing a CV to a JD.
            if token.is_stop:
                continue

            # --- Filter 2: Punctuation and whitespace ---
            # We want to ignore tokens that are purely punctuation or whitespace.
            if token.is_punct or token.is_space:
                continue

            # --- Filter 3: Numeric-only tokens ---
            # Pure numbers like "2019", "123456", "+351" don't help with
            # semantic matching. Note: we keep tokens like "3D" or "Python3"
            # because token.like_num returns False for alphanumeric strings.
            if token.like_num:
                continue

            # --- Filter 4: Minimum length ---
            # Single characters and most 2-character strings (prepositions,
            # abbreviations like "of", "in", "at") carry little semantic value.
            if len(token.text) < 2:
                continue

            # --- Lemmatise and lowercase ---
            # token.lemma_ gives the base form: "running" → "run",
            # "databases" → "database", "required" → "require".
            # We then lowercase to ensure "Python" and "python" are the same token.
            lemma = token.lemma_.lower().strip()

            # Skip any token that becomes empty after stripping
            # (e.g. a token that was only whitespace inside).
            if not lemma:
                continue

            # Unlike in jd_parser.py's _extract_keywords(), we deliberately
            # keep duplicates here. TF-IDF needs to know how many times a term
            # appears (Term Frequency), so removing duplicates would corrupt
            # the TF calculation and reduce the quality of the similarity score.
            tokens.append(lemma)

        return tokens



