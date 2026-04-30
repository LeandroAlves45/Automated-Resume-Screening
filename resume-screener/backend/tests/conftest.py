# =============================================================================
# conftest.py - Fixtures partilhadas dos testes
# =============================================================================
# O pytest carrega este ficheiro automaticamente antes dos testes.
# Fixtures definidas aqui ficam disponíveis em todos os módulos de teste.
# =============================================================================

import tempfile
import pytest
import spacy


@pytest.fixture(scope="session")
def nlp_model():
    """
    Fornece um modelo spaCy inglês vazio para testes dependentes de NLP.

    scope="session" cria o modelo uma única vez, tal como a aplicação faz
    em runtime, e mantém os testes mais rápidos.
    """
    # O modelo vazio suporta tokenização, mas não lematização nem NER.
    return spacy.blank("en")


@pytest.fixture
def sample_jd_text():
    """
    Texto realista de vaga usado por vários módulos de teste.
    """

    return """
    Senior Python Developer

    We are looking for an experienced backend developer to join our team.

    Requirements:
    - Minimum 5 years of experience in backend development
    - Strong proficiency in Python and FastAPI
    - Experience with Docker and PostgreSQL
    - Familiarity with AWS and Git
    - Knowledge of CI/CD pipelines

    Education:
    - Bachelor's degree in Computer Science or related field

    Responsibilities:
    - Design and implement scalable REST APIs
    - Write well-tested and documented Python code
    - Collaborate using Agile methodologies
    """


@pytest.fixture
def strong_cv_text():
    """
    CV que deve pontuar como Strong Match contra a vaga de exemplo.
    """

    return """
    John Smith â€” Senior Python Developer

    EXPERIENCE
    Lead Developer at TechCorp (2019 - 2023)
    Principal Engineer at CloudStartup (2023 - 2026)

    I have 7 years of experience in backend development.

    SKILLS
    Python, FastAPI, Docker, PostgreSQL, AWS, Git, CI/CD

    EDUCATION
    MSc Computer Science, University of Porto, 2018
    """


@pytest.fixture
def weak_cv_text():
    """
    CV que deve pontuar como Weak Match contra a vaga de exemplo.
    """

    return """
    Pedro Alves â€” Junior Designer

    SKILLS
    Figma, Photoshop, Illustrator

    EDUCATION
    High School Certificate, 2021

    I recently finished school and am looking for my first job.
    """


@pytest.fixture
def short_cv_text():
    """
    CV curto demais para conter informação suficiente.
    """

    return "Ana Silva â€” Python Developer"


@pytest.fixture
def tmp_cv_folder(tmp_path, strong_cv_text, weak_cv_text):
    """
    Cria uma pasta temporária com CVs de exemplo para testes de integração.
    """

    # Escreve dois CVs válidos na pasta temporária.
    (tmp_path / "john_smith.txt").write_text(strong_cv_text, encoding="utf-8")
    (tmp_path / "pedro_alves.txt").write_text(weak_cv_text, encoding="utf-8")

    # Ficheiro não suportado que deve ser ignorado silenciosamente.
    (tmp_path / "notes.xyz").write_text("This is not a CV.", encoding="utf-8")

    return tmp_path


@pytest.fixture
def tmp_jd_file(tmp_path, sample_jd_text):
    """
    Cria um ficheiro temporário com a descrição da vaga.
    """

    jd_file = tmp_path / "job_description.txt"
    jd_file.write_text(sample_jd_text, encoding="utf-8")
    return jd_file


@pytest.fixture
def sample_jd_criteria():
    """
    Critérios da vaga já estruturados, como se viessem do JobDescriptionParser.
    """
    return {
        "skills":           ["python", "fastapi", "docker", "postgresql", "aws", "git"],
        "min_experience":   5,
        "education_level":  0.6,
        "keywords":         ["python", "develop", "backend", "api", "docker", "agile"],
        "raw_text":         "Python FastAPI Docker PostgreSQL AWS Git backend development",
    }


@pytest.fixture
def strong_candidate_features():
    """
    Features de um candidato forte, como se viessem do ResumeFeatureExtractor.
    """
    return {
        "matched_skills":   ["python", "fastapi", "docker", "postgresql", "aws", "git"],
        "experience_years": 8,
        "education_level":  0.8,
        "keywords":         ["python","fastapi", "develop", "backend", "aws", "docker", "git"],
        "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
    }


@pytest.fixture
def weak_candidate_features():
    """
    Features de um candidato fraco: sem competências e com experiência mínima.
    """

    return {
        "matched_skills":   [],
        "experience_years": 0,
        "education_level":  0.2,
        "keywords":         ["design","figma", "photoshop"],
        "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
    }
