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
        cv_file.write_text(strong_cv_text, encofing="utf-8")

        # Act: parse the file
        result = parser.parse(str(cv_file))

        # Assert: succesful parse return text and no error
        assert result["error"] is None
        assert len(result["text"]) >= 100
        assert result["name"] == "candidate"
        assert result["file"] == str(cv_file)