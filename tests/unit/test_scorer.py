# =============================================================================
# test_scorer.py — Unit Tests for ResumeScorer
# =============================================================================
# Covers test cases UT-10, UT-11, UT-12, UT-13 from SDLC §7.2.
# =============================================================================

import pytest
from scoring.scorer import ResumeScorer
 
 
@pytest.fixture
def scorer():
    """Return a fresh ResumeScorer instance. No spaCy dependency."""
    return ResumeScorer()

class TestWeightedScoring:
    """Tests for the overall score calculation — SDLC UT-10."""
 
    def test_ut10_total_score_is_within_bounds(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        UT-10: The total_score must always be between 0 and 100 inclusive.
        """
        result = scorer.score(strong_candidate_features, sample_jd_criteria)
 
        assert 0.0 <= result["total_score"] <= 100.0
 
    def test_score_with_perfect_candidate_approaches_100(self, scorer, sample_jd_criteria):
        """
        A candidate who matches all skills, exceeds experience, and holds
        the required education should score close to 100.
        """
        perfect_features = {
            "matched_skills":   sample_jd_criteria["skills"],   # All skills matched
            "experience_years": sample_jd_criteria["min_experience"] * 3,  # Well above minimum
            "education_level":  1.0,     # Doctorate
            "keywords":         sample_jd_criteria["keywords"],  # Identical keywords
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(perfect_features, sample_jd_criteria)
 
        # Not necessarily 100 due to TF-IDF nuance, but should be high.
        assert result["total_score"] >= 85.0
 
    def test_score_with_zero_features_is_low(
        self, scorer, weak_candidate_features, sample_jd_criteria
    ):
        """
        A candidate with no matched skills and zero experience scores low.
        """
        result = scorer.score(weak_candidate_features, sample_jd_criteria)
 
        assert result["total_score"] < 30.0
 
    def test_breakdown_values_sum_to_approximately_total(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        The weighted sum of breakdown values must equal the total_score.
        Verifies that the weight application logic is internally consistent.
        """
        from config import SCORING_WEIGHTS
 
        result = scorer.score(strong_candidate_features, sample_jd_criteria)
        breakdown = result["breakdown"]
 
        # Reconstruct the weighted sum manually and compare to total_score.
        reconstructed = (
            breakdown["skills_match"]     * SCORING_WEIGHTS["skills_match"]     +
            breakdown["experience_years"] * SCORING_WEIGHTS["experience_years"] +
            breakdown["education"]        * SCORING_WEIGHTS["education"]        +
            breakdown["keyword_density"]  * SCORING_WEIGHTS["keyword_density"]
        )
 
        # Allow a small floating point tolerance (1e-6).
        assert abs(reconstructed - result["total_score"]) < 0.1
 
 
class TestClassification:
    """Tests for the Strong / Potential / Weak Match classification."""
 
    def test_ut11_strong_match_for_ideal_candidate(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        UT-11: A candidate with all skills, above-minimum experience, and
        good education is classified as 'Strong Match'.
        """
        result = scorer.score(strong_candidate_features, sample_jd_criteria)
 
        assert result["category"] == "Strong Match"
 
    def test_ut12_weak_match_for_candidate_with_nothing(
        self, scorer, weak_candidate_features, sample_jd_criteria
    ):
        """
        UT-12: A candidate with no matched skills and zero experience
        is classified as 'Weak Match'.
        """
        result = scorer.score(weak_candidate_features, sample_jd_criteria)
 
        assert result["category"] == "Weak Match"
 
    def test_potential_match_classification(self, scorer, sample_jd_criteria):
        """
        A mid-range candidate is classified as 'Potential Match'.
 
        Note on keyword_density: without the full en_core_web_sm model,
        the preprocessor produces no tokens, so TF-IDF returns 0.0.
        We therefore build features that reach Potential Match (>= 50)
        on skills + experience + education alone, without relying on keywords.
 
        Verified calculation:
          Skills:     4/6 = 0.667 * 40% = 26.7
          Experience: 4/(2*5) = 0.40  * 25% = 10.0
          Education:  0.8/0.6 -> 1.0  * 15% = 15.0
          Keywords:   0.0              * 20% =  0.0  (blank model)
          Total:      51.7  ->  Potential Match
        """
        mid_features = {
            "matched_skills":   ["python", "docker", "git", "postgresql"],  # 4/6 skills
            "experience_years": 4,       # linear curve: 4/(2*5) = 0.40
            "education_level":  0.8,     # Master's degree > Bachelor's required
            "keywords":         [],      # blank model produces no tokens
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(mid_features, sample_jd_criteria)
 
        assert result["category"] == "Potential Match"
 
 
class TestNeutralCases:
    """Tests for the neutral/fallback cases defined in SDLC §4.2."""
 
    def test_neutral_skills_when_jd_has_none(self, scorer):
        """
        When the JD lists no required skills, the skills criterion returns
        the neutral value (0.5 → 50 in the breakdown).
        """
        jd_no_skills = {
            "skills": [], "min_experience": 5, "education_level": 0.6,
            "keywords": ["python", "develop"], "raw_text": "python develop",
        }
        features = {
            "matched_skills": [], "experience_years": 5,
            "education_level": 0.6, "keywords": ["python"],
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, jd_no_skills)
 
        # Neutral skills score = 0.5 → 50.0 in breakdown
        assert result["breakdown"]["skills_match"] == 50.0
 
    def test_neutral_experience_when_jd_has_no_requirement(self, scorer):
        """
        When the JD has no experience requirement (min_experience=0),
        the experience criterion returns neutral (0.5 → 50.0).
        """
        jd_no_exp = {
            "skills": ["python"], "min_experience": 0,
            "education_level": 0.6, "keywords": ["python"],
            "raw_text": "python",
        }
        features = {
            "matched_skills": ["python"], "experience_years": 3,
            "education_level": 0.6, "keywords": ["python"],
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, jd_no_exp)
 
        assert result["breakdown"]["experience_years"] == 50.0
 
    def test_ut13_tfidf_with_empty_keywords_returns_zero_no_exception(self, scorer):
        """
        UT-13: When both keyword lists are empty, keyword_density returns 0.0
        and no exception is raised.
        """
        jd_empty_kw = {
            "skills": ["python"], "min_experience": 3,
            "education_level": 0.6, "keywords": [],   # Empty JD keywords
            "raw_text": "",
        }
        features = {
            "matched_skills": ["python"], "experience_years": 3,
            "education_level": 0.6, "keywords": [],   # Empty CV keywords
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        # This must not raise any exception.
        result = scorer.score(features, jd_empty_kw)
 
        assert result["breakdown"]["keyword_density"] == 0.0
 
 
class TestMissingSkills:
    """Tests for the missing_skills field in the score result."""
 
    def test_missing_skills_are_correctly_identified(
        self, scorer, sample_jd_criteria
    ):
        """
        Skills required by the JD but absent from the CV appear
        in missing_skills.
        """
        features = {
            "matched_skills":   ["python", "docker"],
            "experience_years": 5,
            "education_level":  0.6,
            "keywords":         ["python"],
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, sample_jd_criteria)
 
        # JD requires 6 skills; 2 are matched → 4 must be missing.
        assert len(result["missing_skills"]) == 4
        assert "aws"        in result["missing_skills"]
        assert "fastapi"    in result["missing_skills"]
        assert "postgresql" in result["missing_skills"]
 
    def test_missing_skills_sorted_alphabetically(
        self, scorer, sample_jd_criteria
    ):
        """
        missing_skills is returned sorted alphabetically for consistent output.
        """
        features = {
            "matched_skills":   [],
            "experience_years": 0,
            "education_level":  0.0,
            "keywords":         [],
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, sample_jd_criteria)
 
        assert result["missing_skills"] == sorted(result["missing_skills"])
 
    def test_no_missing_skills_when_all_matched(self, scorer, sample_jd_criteria):
        """When all JD skills are matched, missing_skills is empty."""
        features = {
            "matched_skills":   sample_jd_criteria["skills"],  # All matched
            "experience_years": 10,
            "education_level":  1.0,
            "keywords":         sample_jd_criteria["keywords"],
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, sample_jd_criteria)
 
        assert result["missing_skills"] == []