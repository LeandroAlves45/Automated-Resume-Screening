# =============================================================================
# test_jd_parser.py - Testes unitários do JobDescriptionParser
# =============================================================================
# Cobre os casos UT-04, UT-05 e UT-06 definidos na documentação SDLC.
# =============================================================================

import pytest
from backend.parser.jd_parser import JobDescriptionParser


@pytest.fixture
def jd_parser(nlp_model):
    """
    Devolve um JobDescriptionParser com o modelo spaCy partilhado.
    """
    return JobDescriptionParser(nlp_model)


class TestExtractSkills:

    def test_ut04_known_skills_are_detected(self, jd_parser):
        """
        UT-04: uma vaga com 'python' e 'docker' deve extrair ambas.
        """
        jd_text = "We require Python developers with Docker experience."
        result = jd_parser.parse(jd_text)

        assert "python" in result["skills"]
        assert "docker" in result["skills"]

    def test_skills_are_case_insensitive(self, jd_parser):
        """
        A deteção de competências ignora maiúsculas/minúsculas.
        """
        jd_text = "Looking for PYTHON and POSTGRESQL experience."
        result = jd_parser.parse(jd_text)

        assert "python"     in result["skills"]
        assert "postgresql" in result["skills"]

    def test_no_skills_in_jd_returns_empty_list(self, jd_parser):
        """
        Uma vaga sem competências reconhecíveis devolve lista vazia.
        """
        jd_text = "We are looking for a motivated team player."
        result = jd_parser.parse(jd_text)

        assert result["skills"] == []

    def test_skills_list_contains_no_duplicates(self, jd_parser):
        """
        Competências repetidas aparecem apenas uma vez.
        """
        jd_text = "Python is required. Must know Python. Python experience essential."
        result = jd_parser.parse(jd_text)

        # Conta ocorrências de 'python' na lista de competências.
        assert result["skills"].count("python") == 1


class TestExtractExperience:

    def test_ut05_extracts_explicit_years(self, jd_parser):
        """
        UT-05: uma vaga com '3+ years' produz min_experience = 3.
        """
        jd_text = "Requirements: 3+ years of experience in software development."
        result = jd_parser.parse(jd_text)

        assert result["min_experience"] == 3

    def test_ut06_no_experience_requirement_returns_zero(self, jd_parser):
        """
        UT-06: sem menção de anos, min_experience fica 0.
        """
        jd_text = "We are hiring a Python developer. Apply now."
        result = jd_parser.parse(jd_text)

        assert result["min_experience"] == 0

    def test_extracts_minimum_years_pattern(self, jd_parser):
        """O padrão 'Minimum 5 years' é extraído corretamente."""
        result = jd_parser.parse("Minimum 5 years of relevant experience required.")
        assert result["min_experience"] == 5

    def test_extracts_at_least_pattern(self, jd_parser):
        """O padrão 'At least N years' é extraído corretamente."""
        result = jd_parser.parse("You must have at least 4 years of experience.")
        assert result["min_experience"] == 4

    def test_returns_maximum_when_multiple_values_found(self, jd_parser):
        """
        Quando existem vários requisitos, devolve o maior valor.
        """
        jd_text = "Minimum 3 years backend experience. At least 5 years total experience."
        result = jd_parser.parse(jd_text)

        assert result["min_experience"] == 5


class TestExtractEducation:

    def test_bachelor_requirement_returns_correct_value(self, jd_parser):
        """Uma vaga com bachelor's degree devolve education_level = 0.6."""
        result = jd_parser.parse("Bachelor's degree in Computer Science required.")
        assert result["education_level"] == 0.6

    def test_master_requirement_returns_correct_value(self, jd_parser):
        """Uma vaga com master's degree devolve education_level = 0.8."""
        result = jd_parser.parse("Master's degree preferred.")
        assert result["education_level"] == 0.8

    def test_no_education_requirement_returns_zero(self, jd_parser):
        """Sem requisito de formação, devolve education_level = 0.0."""
        result = jd_parser.parse("Looking for a skilled Python developer.")
        assert result["education_level"] == 0.0

    def test_highest_level_wins_when_multiple_mentioned(self, jd_parser):
        """
        Se Bachelor e Master aparecem, vence o nível mais alto.
        """
        jd_text = "Bachelor's required. Master's preferred."
        result = jd_parser.parse(jd_text)
        assert result["education_level"] == 0.8


class TestParseOutputStructure:

    def test_parse_returns_all_required_keys(self, jd_parser, sample_jd_text):
        """parse() deve devolver sempre as cinco chaves esperadas."""
        result = jd_parser.parse(sample_jd_text)

        assert "skills"          in result
        assert "min_experience"  in result
        assert "education_level" in result
        assert "keywords"        in result
        assert "raw_text"        in result

    def test_empty_jd_returns_safe_defaults(self, jd_parser):
        """Uma vaga vazia devolve valores seguros por defeito."""
        result = jd_parser.parse("")

        assert result["skills"]          == []
        assert result["min_experience"]  == 0
        assert result["education_level"] == 0.0
        assert result["keywords"]        == []

    def test_raw_text_preserved_in_output(self, jd_parser, sample_jd_text):
        """O texto original da vaga é preservado em raw_text para TF-IDF."""
        result = jd_parser.parse(sample_jd_text)
        assert result["raw_text"] == sample_jd_text
