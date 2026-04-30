# =============================================================================
# test_extractor.py - Testes unitários do ResumeFeatureExtractor
# =============================================================================
# Cobre os casos UT-07, UT-08 e UT-09 definidos na documentação SDLC.
# =============================================================================

import pytest
from datetime import datetime
from backend.nlp.extractor import ResumeFeatureExtractor


@pytest.fixture
def extractor(nlp_model):
    """Devolve um ResumeFeatureExtractor com o modelo spaCy partilhado."""
    return ResumeFeatureExtractor(nlp_model)


@pytest.fixture
def jd_skills():
    """Conjunto padrão de competências da vaga usado nestes testes."""
    return ["python", "fastapi", "docker", "postgresql", "aws", "git"]


class TestMatchedSkills:
    """Testes de correspondência de competências."""

    def test_ut07_skills_present_in_cv_are_matched(self, extractor, jd_skills):
        """
        UT-07: um CV com 'python' inclui 'python' em matched_skills.
        """
        cv_text = "I am experienced in Python, Docker, and Git."
        result = extractor.extract(cv_text, jd_skills)

        assert "python" in result["matched_skills"]
        assert "docker" in result["matched_skills"]
        assert "git"    in result["matched_skills"]

    def test_skills_not_in_cv_are_not_matched(self, extractor, jd_skills):
        """Competências ausentes no CV não devem ser marcadas como encontradas."""
        cv_text = "I work exclusively with Java and Spring Boot."
        result = extractor.extract(cv_text, jd_skills)

        assert "python"     not in result["matched_skills"]
        assert "docker"     not in result["matched_skills"]
        assert "postgresql" not in result["matched_skills"]

    def test_skill_matching_is_case_insensitive(self, extractor, jd_skills):
        """A correspondência ignora diferenças de maiúsculas/minúsculas."""
        cv_text = "PYTHON developer with DOCKER experience."
        result = extractor.extract(cv_text, jd_skills)

        assert "python" in result["matched_skills"]
        assert "docker" in result["matched_skills"]

    def test_empty_jd_skills_returns_empty_matched(self, extractor):
        """Sem competências na vaga, matched_skills deve ficar vazio."""
        cv_text = "Python developer with 5 years of experience."
        result = extractor.extract(cv_text, jd_skills=[])

        assert result["matched_skills"] == []


class TestExperienceExtraction:
    """Testes de extração de experiência."""

    def test_ut08_explicit_phrase_extracted(self, extractor, jd_skills):
        """
        UT-08: uma frase com '5 years experience' devolve 5 anos.
        """
        cv_text = "I have 5 years of experience in backend development."
        result = extractor.extract(cv_text, jd_skills)

        assert result["experience_years"] == 5

    def test_ut09_date_ranges_summed_correctly(self, extractor, jd_skills):
        """
        UT-09: intervalo '2019 - 2023' produz pelo menos 4 anos.
        """
        cv_text = """
        Software Engineer at Company A (2019 - 2023)
        Python development and API design.
        """
        result = extractor.extract(cv_text, jd_skills)

        assert result["experience_years"] >= 4

    def test_multiple_date_ranges_are_summed(self, extractor, jd_skills):
        """
        Períodos profissionais múltiplos são somados.
        """
        cv_text = """
        Developer at Company A (2018 - 2021)
        Senior Developer at Company B (2021 - 2025)
        """
        result = extractor.extract(cv_text, jd_skills)

        assert result["experience_years"] == 7

    def test_present_keyword_uses_current_year(self, extractor, jd_skills):
        """
        'Present' num intervalo de datas é tratado como o ano atual.
        """
        current_year = datetime.now().year
        start_year = 2020
        expected_min = current_year - start_year

        cv_text = f"Lead Engineer at Corp ({start_year} - Present)"
        result = extractor.extract(cv_text, jd_skills)

        assert result["experience_years"] >= expected_min

    def test_takes_maximum_of_both_strategies(self, extractor, jd_skills):
        """
        Quando as duas estratégias divergem, é usado o maior valor.
        """
        cv_text = """
        Developer (2015 - 2022)
        I have 3 years of experience.
        """
        result = extractor.extract(cv_text, jd_skills)

        # O intervalo dá 7 e a frase explícita dá 3; o resultado deve ser 7.
        assert result["experience_years"] == 7

    def test_no_experience_mentioned_returns_zero(self, extractor, jd_skills):
        """Sem datas ou frases de experiência, devolve 0."""
        cv_text = "Python developer. Skills: Docker, PostgreSQL, Git."
        result = extractor.extract(cv_text, jd_skills)

        assert result["experience_years"] == 0


class TestEducationExtraction:
    """Testes de extração do nível de formação."""

    def test_bachelor_detected_in_cv(self, extractor, jd_skills):
        """'BSc' no CV produz education_level = 0.6."""
        cv_text = "BSc Computer Science, University of Porto, 2018. " * 5
        result = extractor.extract(cv_text, jd_skills)

        assert result["education_level"] == 0.6

    def test_master_detected_in_cv(self, extractor, jd_skills):
        """'Master' no CV produz education_level = 0.8."""
        cv_text = "Master of Science in Software Engineering, 2020. " * 5
        result = extractor.extract(cv_text, jd_skills)

        assert result["education_level"] == 0.8

    def test_highest_level_returned_when_multiple_present(self, extractor, jd_skills):
        """
        Se Bachelor e Master aparecem, é devolvido o nível mais alto.
        """
        cv_text = "Bachelor 2016. Master 2018. Python developer. " * 3
        result = extractor.extract(cv_text, jd_skills)

        assert result["education_level"] == 0.8

    def test_no_education_returns_zero(self, extractor, jd_skills):
        """Sem palavras-chave de formação, devolve 0.0."""
        cv_text = "Python developer with 5 years experience in Docker and AWS."
        result = extractor.extract(cv_text, jd_skills)

        assert result["education_level"] == 0.0


class TestExtractOutputStructure:
    """Testes da estrutura devolvida por extract()."""

    def test_extract_returns_all_required_keys(self, extractor, strong_cv_text, jd_skills):
        """extract() deve devolver sempre as cinco chaves esperadas."""
        result = extractor.extract(strong_cv_text, jd_skills)

        assert "matched_skills"   in result
        assert "experience_years" in result
        assert "education_level"  in result
        assert "keywords"         in result
        assert "entities"         in result

    def test_empty_cv_returns_zero_features(self, extractor, jd_skills):
        """Um CV vazio devolve features zeradas, sem exceção."""
        result = extractor.extract("", jd_skills)

        assert result["matched_skills"]   == []
        assert result["experience_years"] == 0
        assert result["education_level"]  == 0.0

    def test_entities_dict_has_all_four_keys(self, extractor, strong_cv_text, jd_skills):
        """O dicionário de entidades mantém ORG, GPE, DATE e PERSON."""
        result = extractor.extract(strong_cv_text, jd_skills)

        assert "ORG"    in result["entities"]
        assert "GPE"    in result["entities"]
        assert "DATE"   in result["entities"]
        assert "PERSON" in result["entities"]
