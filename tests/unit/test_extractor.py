# =============================================================================
# test_extractor.py — Unit Tests for ResumeFeatureExtractor
# =============================================================================
# Covers test cases UT-07, UT-08, UT-09 from SDLC §7.2.
# =============================================================================
 
import pytest
from datetime import datetime
from nlp.extractor import ResumeFeatureExtractor

@pytest.fixture
def extractor(nlp_model):
    """Return a ResumeFeatureExtractor using the shared blank spaCy model."""
    return ResumeFeatureExtractor(nlp_model)

@pytest.fixture
def jd_skills():
    """Standard set of JD skills used across extractor tests."""
    return ["python", "fastapi", "docker", "postgresql", "aws", "git"]

class TestMatchedSkills:
    """Tests for skill matching — SDLC UT-07."""
 
    def test_ut07_skills_present_in_cv_are_matched(self, extractor, jd_skills):
        """
        UT-07: A CV containing 'python' produces matched_skills
        that includes 'python'.
        """
        cv_text = "I am experienced in Python, Docker, and Git."
        result = extractor.extract(cv_text, jd_skills)
 
        assert "python" in result["matched_skills"]
        assert "docker" in result["matched_skills"]
        assert "git"    in result["matched_skills"]

    def test_skills_not_in_cv_are_not_matched(self, extractor, jd_skills):
        """Skills that appear in the JD but not the CV must not be matched."""
        cv_text = "I work exclusively with Java and Spring Boot."
        result = extractor.extract(cv_text, jd_skills)
 
        assert "python"     not in result["matched_skills"]
        assert "docker"     not in result["matched_skills"]
        assert "postgresql" not in result["matched_skills"]
 
    def test_skill_matching_is_case_insensitive(self, extractor, jd_skills):
        """Skill matching works regardless of capitalisation in the CV."""
        cv_text = "PYTHON developer with DOCKER experience."
        result = extractor.extract(cv_text, jd_skills)
 
        assert "python" in result["matched_skills"]
        assert "docker" in result["matched_skills"]
 
    def test_empty_jd_skills_returns_empty_matched(self, extractor):
        """When the JD has no skills, matched_skills must be empty."""
        cv_text = "Python developer with 5 years of experience."
        result = extractor.extract(cv_text, jd_skills=[])
 
        assert result["matched_skills"] == []

class TestExperienceExtraction:
    """Tests for experience extraction — SDLC UT-08, UT-09."""
 
    def test_ut08_explicit_phrase_extracted(self, extractor, jd_skills):
        """
        UT-08: A CV with '5 years experience' produces experience_years = 5.
        """
        cv_text = "I have 5 years of experience in backend development."
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["experience_years"] == 5
 
    def test_ut09_date_ranges_summed_correctly(self, extractor, jd_skills):
        """
        UT-09: A CV with date ranges '2019 - 2023' produces
        experience_years >= 4.
        """
        cv_text = """
        Software Engineer at Company A (2019 - 2023)
        Python development and API design.
        """
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["experience_years"] >= 4
 
    def test_multiple_date_ranges_are_summed(self, extractor, jd_skills):
        """
        Multiple employment periods are summed together.
        2018-2021 (3yr) + 2021-2025 (4yr) = 7yr total.
        """
        cv_text = """
        Developer at Company A (2018 - 2021)
        Senior Developer at Company B (2021 - 2025)
        """
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["experience_years"] == 7
 
    def test_present_keyword_uses_current_year(self, extractor, jd_skills):
        """
        'Present' in a date range is treated as the current year.
        """
        current_year = datetime.now().year
        start_year = 2020
        expected_min = current_year - start_year
 
        cv_text = f"Lead Engineer at Corp ({start_year} - Present)"
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["experience_years"] >= expected_min
 
    def test_takes_maximum_of_both_strategies(self, extractor, jd_skills):
        """
        When explicit phrase gives less than date ranges, the higher
        value (from date ranges) is returned.
        """
        cv_text = """
        Developer (2015 - 2022)
        I have 3 years of experience.
        """
        result = extractor.extract(cv_text, jd_skills)
 
        # Date range gives 7, explicit gives 3 — result must be 7.
        assert result["experience_years"] == 7
 
    def test_no_experience_mentioned_returns_zero(self, extractor, jd_skills):
        """A CV with no dates or experience phrases returns 0."""
        cv_text = "Python developer. Skills: Docker, PostgreSQL, Git."
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["experience_years"] == 0

class TestEducationExtraction:
    """Tests for education level extraction from CVs."""
 
    def test_bachelor_detected_in_cv(self, extractor, jd_skills):
        """'BSc' in a CV produces education_level = 0.6."""
        cv_text = "BSc Computer Science, University of Porto, 2018. " * 5
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["education_level"] == 0.6
 
    def test_master_detected_in_cv(self, extractor, jd_skills):
        """'Master' in a CV produces education_level = 0.8."""
        cv_text = "Master of Science in Software Engineering, 2020. " * 5
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["education_level"] == 0.8
 
    def test_highest_level_returned_when_multiple_present(self, extractor, jd_skills):
        """
        A CV mentioning both Bachelor and Master returns the higher value.
        A candidate who lists both degrees should be credited with the highest.
        """
        cv_text = "Bachelor 2016. Master 2018. Python developer. " * 3
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["education_level"] == 0.8
 
    def test_no_education_returns_zero(self, extractor, jd_skills):
        """A CV with no education keywords returns education_level = 0.0."""
        cv_text = "Python developer with 5 years experience in Docker and AWS."
        result = extractor.extract(cv_text, jd_skills)
 
        assert result["education_level"] == 0.0
 
 
class TestExtractOutputStructure:
    """Tests that extract() always returns the correct dict structure."""
 
    def test_extract_returns_all_required_keys(self, extractor, strong_cv_text, jd_skills):
        """extract() must always return all five expected keys."""
        result = extractor.extract(strong_cv_text, jd_skills)
 
        assert "matched_skills"   in result
        assert "experience_years" in result
        assert "education_level"  in result
        assert "keywords"         in result
        assert "entities"         in result
 
    def test_empty_cv_returns_zero_features(self, extractor, jd_skills):
        """An empty CV string returns zero-value features, not an exception."""
        result = extractor.extract("", jd_skills)
 
        assert result["matched_skills"]   == []
        assert result["experience_years"] == 0
        assert result["education_level"]  == 0.0
 
    def test_entities_dict_has_all_four_keys(self, extractor, strong_cv_text, jd_skills):
        """The entities dict always has ORG, GPE, DATE, and PERSON keys."""
        result = extractor.extract(strong_cv_text, jd_skills)
 
        assert "ORG"    in result["entities"]
        assert "GPE"    in result["entities"]
        assert "DATE"   in result["entities"]
        assert "PERSON" in result["entities"]

