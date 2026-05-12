# =============================================================================
# test_resume_parser.py — Unit Tests for ResumeParser
# =============================================================================
# Covers test cases UT-01, UT-02, UT-03 from SDLC §7.2.
#
# Testing strategy for this module:
# ResumeParser has no external NLP dependencies — it only uses pdfminer,
# python-docx, and standard library. All tests use real file I/O via
# pytest's tmp_path fixture, which creates actual files on disk.
# =============================================================================

from unittest import result

import pytest
from parser.resume_parser import ResumeParser

# ===============================================================================
# FIXTURES local to this test module
# ===============================================================================

@pytest.fixture
def parser():
    return ResumeParser()

# ===============================================================================
# UNIT TESTS - parse() method
# ===============================================================================

class TestParseSingleFile:

    def test_ut01_valid_txt_file_returns_text(self, parser, tmp_path, strong_cv_text):
        """
        UT-01: Parsing a valid TXT file returns a dict with text populated
        and error set to None.
        """

        # Arrange: write a real life to the temp directory
        cv_file = tmp_path / "candidate.txt"
        cv_file.write_text(strong_cv_text, encoding="utf-8")

        # Act: parse the file
        result = parser.parse(str(cv_file))

        # Assert: succesful parse return text and no error
        assert result["error"] is None
        assert len(result["text"]) >= 100
        assert result["name"] == "candidate"
        assert result["file"] == str(cv_file)

    def test_ut02_nonexistent_file_returns_error(self, parser, tmp_path):
        """
        UT-02: Parsing a file that does not exist returns a dict with
        error populated and text empty. Must NOT raise an exception.
        """
        # Arrange: path to a file that does not exist
        missing_path = str(tmp_path / "ghost.txt")

        # Act
        result = parser.parse(missing_path)

        # Assert: error is populated, text is empty
        assert result["error"] is not None
        assert "not found" in result["error"].lower()
        assert result["text"] == ""

    def test_ut03_cv_too_short_returns_error(self, parser, tmp_path, short_cv_text):
        """
        UT-03: A CV with fewer than 100 characters after extraction is
        considered invalid — error is set and text is discarded.
        """
        # Arrange: write a file with very short content
        cv_file = tmp_path / "tiny.txt"
        cv_file.write_text(short_cv_text, encoding="utf-8")

        # Act
        result = parser.parse(str(cv_file))

        # Assert: error mentions the lenght issue
        assert result["error"] is not None
        assert "too short" in result["error"].lower()
        assert result["text"] == ""

    def test_unsupported_extension_returns_error(self, parser, tmp_path):
        """
        Parsing a file with an unsupported extension returns an error
        and does NOT attempt to read the file.
        """

        xyz_file = tmp_path / "document.xyz"
        xyz_file.write_text("Some content", encoding="utf-8")

        result = parser.parse(str(xyz_file))

        assert result["error"] is not None
        assert "unsupported" in result["error"].lower()

    def test_result_dict_always_has_all_keys(self, parser, tmp_path):
        """
        The result dict must always contain all four keys — even on failure.
        This guarantees the caller never gets a KeyError.
        """
        # Use a missing file to trigger failure path
        result = parser.parse(str(tmp_path / "missing.pdf"))

        # All four keys must be present
        assert "name" in result
        assert "file" in result
        assert "text" in result
        assert "error" in result

    def test_txt_encoding_fallback_latin1(self, parser, tmp_path):
        """
        A TXT file with Latin-1 encoding is read correctly via the
        encoding fallback (NFR-09).
        """
        # Write a file in Latin-1 encoding with accented characters
        cv_file = tmp_path / "latin_cv.txt"
        content = "João Silva\nSenior Developer\n" + "Python experience." * 10
        cv_file.write_bytes(content.encode("latin-1"))

        result = parser.parse(str(cv_file))

        # The file should parse without error despite non-UTF-8 encoding
        assert result["error"] is None
        assert "Jo" in result["text"]  # Name was extracted

# ===============================================================================
# UNIT TESTS - parse() method
# ===============================================================================

class TestParseFolder:

    def test_folder_returns_pnly_supported_files(self, parser, tmp_cv_folder):
        """
        parse_folder() returns results only for supported extensions.
        The .xyz file in tmp_cv_folder must be silently skipped.
        """
        results = parser.parse_folder(str(tmp_cv_folder))

        # Only .txt and .pdf files should be processed
        assert len(results) == 2
        
    def test_folder_nonexistent_returns_empty_list(self, parser, tmp_path):
        """
        parse_folder() on a path that doesn't exist returns an empty list,
        not an exception.
        """

        results = parser.parse_folder(str(tmp_path / "no_such_folder"))
        assert results == []

    def test_folder_results_sorted_alphabetically(self, parser, tmp_cv_folder):
        """
        Results from parse_folder() are returned in alphabetical order
        by filename (consistent cross-platform ordering).
        """
        results = parser.parse_folder(str(tmp_cv_folder))
        names = [r["name"] for r in results]

        # Names should be in alphabetical order
        assert names == sorted(names)

    def test_bad_file_does_not_abort_batch(self, parser, tmp_path, strong_cv_text):
        """
        A file that fails to parse (too short) does not prevent other
        files in the same folder from being processed (NFR-06: Robustness).
        """

        # Write one valid CV
        (tmp_path / "valid_cando.txt").write_text(strong_cv_text, encoding="utf-8")
        # Write one invalid CV (too short)
        (tmp_path / "empty_cv.txt").write_text("Too short", encoding="utf-8")

        results = parser.parse_folder(str(tmp_path))

        # Both files are attempted - we get 2 results
        assert len(results) == 2

        # Exactly one succeds and one fails
        successes = [r for r in results if r["error"] is None]
        failures = [r for r in results if r["error"] is not None]
        assert len(successes) == 1
        assert len(failures) == 1