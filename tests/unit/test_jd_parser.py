# =============================================================================
# test_jd_parser.py — Unit Tests for JobDescriptionParser
# =============================================================================
# Covers test cases UT-04, UT-05, UT-06 from SDLC.
# =============================================================================


import pytest
from parser.jd_parser import JobDescriptionParser

@pytest.fixture
def jd_parser(nlp_model):
    """
    Return a JobDescriptionParser instance using the shared blank spaCy model.
    The nlp_model fixture is defined in conftest.py and injected by pytest.
    """
    return JobDescriptionParser(nlp_model)

class TestExtractSkills:
 
    def test_ut04_known_skills_are_detected(self, jd_parser):
        """
        UT-04: A JD containing 'python' and 'docker' must have both
        in the extracted skills list.
        """
        jd_text = "We require Python developers with Docker experience."
        result = jd_parser.parse(jd_text)
 
        assert "python" in result["skills"]
        assert "docker" in result["skills"]
 
    def test_skills_are_case_insensitive(self, jd_parser):
        """
        Skill detection must work regardless of capitalisation in the JD.
        """
        jd_text = "Looking for PYTHON and POSTGRESQL experience."
        result = jd_parser.parse(jd_text)
 
        assert "python"     in result["skills"]
        assert "postgresql" in result["skills"]

    def test_no_skills_in_jd_returns_empty_list(self, jd_parser):
        """
        A JD with no recognisable technical skills returns an empty list,
        not None or an error.
        """
        jd_text = "We are looking for a motivated team player."
        result = jd_parser.parse(jd_text)
 
        assert result["skills"] == []
 
    def test_skills_list_contains_no_duplicates(self, jd_parser):
        """
        If a skill is mentioned multiple times, it should appear only once.
        """
        jd_text = "Python is required. Must know Python. Python experience essential."
        result = jd_parser.parse(jd_text)
 
        # Count occurrences of 'python' in the skills list.
        assert result["skills"].count("python") == 1

class TestExtractExperience:
 
    def test_ut05_extracts_explicit_years(self, jd_parser):
        """
        UT-05: A JD with '3+ years' produces min_experience = 3.
        """
        jd_text = "Requirements: 3+ years of experience in software development."
        result = jd_parser.parse(jd_text)
 
        assert result["min_experience"] == 3
 
    def test_ut06_no_experience_requirement_returns_zero(self, jd_parser):
        """
        UT-06: A JD with no mention of years returns min_experience = 0.
        """
        jd_text = "We are hiring a Python developer. Apply now."
        result = jd_parser.parse(jd_text)
 
        assert result["min_experience"] == 0
 
    def test_extracts_minimum_years_pattern(self, jd_parser):
        """'Minimum 5 years' pattern is extracted correctly."""
        result = jd_parser.parse("Minimum 5 years of relevant experience required.")
        assert result["min_experience"] == 5

    def test_extracts_at_least_pattern(self, jd_parser):
        """'At least N years' pattern is extracted correctly."""
        result = jd_parser.parse("You must have at least 4 years of experience.")
        assert result["min_experience"] == 4
 
    def test_returns_maximum_when_multiple_values_found(self, jd_parser):
        """
        When multiple experience requirements appear, the strictest
        (highest) value is returned.
        """
        jd_text = "Minimum 3 years backend experience. At least 5 years total experience."
        result = jd_parser.parse(jd_text)
 
        assert result["min_experience"] == 5

class TestExtractEducation:
 
    def test_bachelor_requirement_returns_correct_value(self, jd_parser):
        """A JD requiring a bachelor's degree returns education_level = 0.6."""
        result = jd_parser.parse("Bachelor's degree in Computer Science required.")
        assert result["education_level"] == 0.6
 
    def test_master_requirement_returns_correct_value(self, jd_parser):
        """A JD requiring a master's degree returns education_level = 0.8."""
        result = jd_parser.parse("Master's degree preferred.")
        assert result["education_level"] == 0.8
 
    def test_no_education_requirement_returns_zero(self, jd_parser):
        """A JD with no education mention returns education_level = 0.0."""
        result = jd_parser.parse("Looking for a skilled Python developer.")
        assert result["education_level"] == 0.0
 
    def test_highest_level_wins_when_multiple_mentioned(self, jd_parser):
        """
        If a JD mentions both Bachelor and Master, the higher level (0.8)
        should be returned.
        """
        jd_text = "Bachelor's required. Master's preferred."
        result = jd_parser.parse(jd_text)
        assert result["education_level"] == 0.8
 
 
class TestParseOutputStructure:
 
    def test_parse_returns_all_required_keys(self, jd_parser, sample_jd_text):
        """parse() must always return all five expected keys."""
        result = jd_parser.parse(sample_jd_text)
 
        assert "skills"          in result
        assert "min_experience"  in result
        assert "education_level" in result
        assert "keywords"        in result
        assert "raw_text"        in result
 
    def test_empty_jd_returns_safe_defaults(self, jd_parser):
        """An empty JD string returns safe zero-value defaults, not an error."""
        result = jd_parser.parse("")
 
        assert result["skills"]          == []
        assert result["min_experience"]  == 0
        assert result["education_level"] == 0.0
        assert result["keywords"]        == []
 
    def test_raw_text_preserved_in_output(self, jd_parser, sample_jd_text):
        """The original JD text is preserved in raw_text for TF-IDF use."""
        result = jd_parser.parse(sample_jd_text)
        assert result["raw_text"] == sample_jd_text