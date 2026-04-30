# =============================================================================
# test_scorer.py - Testes unitários do ResumeScorer
# =============================================================================
# Cobre os casos UT-10, UT-11, UT-12 e UT-13 da documentação SDLC.
# =============================================================================

import pytest
from backend.scoring.scorer import ResumeScorer


@pytest.fixture
def scorer():
    """Devolve uma nova instância de ResumeScorer."""
    return ResumeScorer()


class TestWeightedScoring:
    """Testes do cálculo de pontuação global."""

    def test_ut10_total_score_is_within_bounds(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        UT-10: total_score deve ficar sempre entre 0 e 100.
        """
        result = scorer.score(strong_candidate_features, sample_jd_criteria)

        assert 0.0 <= result["total_score"] <= 100.0

    def test_score_with_perfect_candidate_approaches_100(self, scorer, sample_jd_criteria):
        """
        Um candidato ideal deve aproximar-se de 100 pontos.
        """
        perfect_features = {
            "matched_skills":   sample_jd_criteria["skills"],   # Todas as competências encontradas
            "experience_years": sample_jd_criteria["min_experience"] * 3,  # Muito acima do mínimo
            "education_level":  1.0,     # Doutoramento
            "keywords":         sample_jd_criteria["keywords"],  # Palavras-chave idênticas
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(perfect_features, sample_jd_criteria)

        # Pode não ser 100 por detalhes do TF-IDF, mas deve ser alto.
        assert result["total_score"] >= 85.0

    def test_score_with_zero_features_is_low(
        self, scorer, weak_candidate_features, sample_jd_criteria
    ):
        """
        Candidato sem competências e sem experiência pontua baixo.
        """
        result = scorer.score(weak_candidate_features, sample_jd_criteria)

        assert result["total_score"] < 30.0

    def test_breakdown_values_sum_to_approximately_total(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        A soma ponderada do breakdown deve aproximar-se do total_score.
        """
        from backend.api.scoring_config import SCORING_WEIGHTS

        result = scorer.score(strong_candidate_features, sample_jd_criteria)
        breakdown = result["breakdown"]

        # Reconstrói manualmente a soma ponderada.
        reconstructed = (
            breakdown["skills_match"]     * SCORING_WEIGHTS["skills_match"]     +
            breakdown["experience_years"] * SCORING_WEIGHTS["experience_years"] +
            breakdown["education"]        * SCORING_WEIGHTS["education"]        +
            breakdown["keyword_density"]  * SCORING_WEIGHTS["keyword_density"]
        )

        # Pequena tolerância para arredondamentos de floats.
        assert abs(reconstructed - result["total_score"]) < 0.1


class TestClassification:
    """Testes das categorias Strong, Potential e Weak Match."""

    def test_ut11_strong_match_for_ideal_candidate(
        self, scorer, strong_candidate_features, sample_jd_criteria
    ):
        """
        UT-11: candidato forte é classificado como Strong Match.
        """
        result = scorer.score(strong_candidate_features, sample_jd_criteria)

        assert result["category"] == "Strong Match"

    def test_ut12_weak_match_for_candidate_with_nothing(
        self, scorer, weak_candidate_features, sample_jd_criteria
    ):
        """
        UT-12: candidato sem competências nem experiência é Weak Match.
        """
        result = scorer.score(weak_candidate_features, sample_jd_criteria)

        assert result["category"] == "Weak Match"

    def test_potential_match_classification(self, scorer, sample_jd_criteria):
        """
        Candidato intermédio é classificado como Potential Match.

        Como o modelo spaCy vazio não produz tokens úteis, este caso atinge
        Potential Match através de competências, experiência e formação.
        """
        mid_features = {
            "matched_skills":   ["python", "docker", "git", "postgresql"],  # 4/6 competências
            "experience_years": 4,       # curva linear: 4/(2*5) = 0.40
            "education_level":  0.8,     # Mestrado acima de licenciatura exigida
            "keywords":         [],      # modelo vazio não produz tokens
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(mid_features, sample_jd_criteria)

        assert result["category"] == "Potential Match"


class TestNeutralCases:
    """Testes dos casos neutros e de fallback."""

    def test_neutral_skills_when_jd_has_none(self, scorer):
        """
        Sem competências na vaga, o critério de skills devolve valor neutro.
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

        # Score neutro de skills = 0.5 -> 50.0 no breakdown.
        assert result["breakdown"]["skills_match"] == 50.0

    def test_neutral_experience_when_jd_has_no_requirement(self, scorer):
        """
        Sem requisito mínimo, experiência devolve valor neutro.
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
        UT-13: listas vazias de keywords devolvem 0.0 sem exceção.
        """
        jd_empty_kw = {
            "skills": ["python"], "min_experience": 3,
            "education_level": 0.6, "keywords": [],   # Keywords da vaga vazias
            "raw_text": "",
        }
        features = {
            "matched_skills": ["python"], "experience_years": 3,
            "education_level": 0.6, "keywords": [],   # Keywords do CV vazias
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        # Este caso não deve lançar exceção.
        result = scorer.score(features, jd_empty_kw)

        assert result["breakdown"]["keyword_density"] == 0.0


class TestMissingSkills:
    """Testes do campo missing_skills."""

    def test_missing_skills_are_correctly_identified(
        self, scorer, sample_jd_criteria
    ):
        """
        Competências exigidas e ausentes no CV aparecem em missing_skills.
        """
        features = {
            "matched_skills":   ["python", "docker"],
            "experience_years": 5,
            "education_level":  0.6,
            "keywords":         ["python"],
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, sample_jd_criteria)

        # A vaga exige 6 competências; 2 foram encontradas; 4 ficam em falta.
        assert len(result["missing_skills"]) == 4
        assert "aws"        in result["missing_skills"]
        assert "fastapi"    in result["missing_skills"]
        assert "postgresql" in result["missing_skills"]

    def test_missing_skills_sorted_alphabetically(
        self, scorer, sample_jd_criteria
    ):
        """
        missing_skills é devolvido por ordem alfabética.
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
        """Quando todas as competências são encontradas, missing_skills fica vazio."""
        features = {
            "matched_skills":   sample_jd_criteria["skills"],  # Todas encontradas
            "experience_years": 10,
            "education_level":  1.0,
            "keywords":         sample_jd_criteria["keywords"],
            "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
        }
        result = scorer.score(features, sample_jd_criteria)

        assert result["missing_skills"] == []
