# =============================================================================
# conftest.py — Shared Test Fixtures
# =============================================================================
# pytest automatically discovers and loads this file before running any tests.
# Fixtures defined here are available to ALL test files in the project without
# needing to import them explicitly.
#
# A fixture is a reusable piece of test setup — think of it as a function
# that creates a known, controlled state before a test runs. Instead of
# repeating setup code in every test, you declare the fixture as a parameter
# and pytest injects it automatically.
#
# Example:
#   def test_something(nlp_model, sample_jd_text):
#       # nlp_model and sample_jd_text are injected by pytest from this file.
#       ...
# =============================================================================

import tempfile
import pytest
import spacy

# =============================================================================
# NLP MODEL FIXTURE
# =============================================================================

@pytest.fixture(scope="session")
def nlp_model():
    """
    Provide a spaCy blank English model for all NLP-dependent tests.
 
    scope="session" means this fixture is created ONCE for the entire test
    session and shared across all tests that request it. This mirrors how
    the real application works — the model is loaded once and injected
    everywhere. It also keeps tests fast: loading spaCy even as a blank
    model has a small overhead that adds up across many tests.
 
    In a real environment with en_core_web_sm installed, you would change
    this to: return spacy.load("en_core_web_sm")
 
    Returns:
        spacy.Language: A blank English language model for testing.
    """
    # We use spacy.blank("en") instead of spacy.load("en_core_web_sm")
    # because the full model is not available in this environment.
    # The blank model supports tokenisation but NOT lemmatisation or NER.
    # This means token-level tests will work; lemma-specific assertions
    # will need to account for the fact that lemma_ == text in a blank model.
    return spacy.blank("en")

# =============================================================================
# TEXT CONTENT FIXTURE
# =============================================================================

@pytest.fixture
def sample_jd_text():
    """
    A realistic Job Description text for use across multiple test modules.
 
    Contains all the elements the JD parser needs to extract:
    skills, experience requirement, education requirement, and keywords.
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
    A CV text that should score as a Strong Match against the sample JD.
 
    Contains many matching skills, sufficient experience, and relevant education.
    """

    return """
    John Smith — Senior Python Developer
 
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
    A CV text that should score as a Weak Match against the sample JD.
 
    Contains no matching skills, minimal experience, and low education level.
    """

    return """
    Pedro Alves — Junior Designer
 
    SKILLS
    Figma, Photoshop, Illustrator
 
    EDUCATION
    High School Certificate, 2021
 
    I recently finished school and am looking for my first job.
    """

@pytest.fixture
def short_cv_text():
    """
    A CV text that is too short to contain all required information.
 
    This tests how the system handles incomplete data.
    """

    return "Ana Silva — Python Developer"

# =============================================================================
# FILE SYSTEM FIXTURES
# =============================================================================

@pytest.fixture
def tmp_cv_folder(tmp_path, strong_cv_text, weak_cv_text):
    """
    Create a temporary folder with sample CV files for integration tests.
 
    pytest's built-in tmp_path fixture provides a temporary directory that
    is automatically cleaned up after the test completes. We use it to
    create real files on disk without polluting the project directory.
 
    Args:
        tmp_path: Built-in pytest fixture — a pathlib.Path to a temp dir.
        strong_cv_text: The strong candidate CV fixture defined above.
        weak_cv_text: The weak candidate CV fixture defined above.
 
    Returns:
        pathlib.Path: Path to the temporary folder containing CV files.
    """

    # Write two valid CV files to the temp folder
    (tmp_path / "john_smith.txt").write_text(strong_cv_text, encoding="utf-8")
    (tmp_path / "pedro_alves.txt").write_text(weak_cv_text, encoding="utf-8")

    # Write one unsupported file - should be silently ignored by the system
    (tmp_path / "notes.xyz").write_text("This is not a CV.", encoding="utf-8")


    return tmp_path

@pytest.fixture
def tmp_jd_file(tmp_path, sample_jd_text):
    """
    Create a temporary JD text file for integration tests.
 
    Args:
        tmp_path: Built-in pytest fixture.
        sample_jd_text: The sample JD fixture defined above.
 
    Returns:
        pathlib.Path: Path to the temporary JD file.
    """

    jd_file = tmp_path / "job_description.txt"
    jd_file.write_text(sample_jd_text, encoding="utf-8")
    return jd_file

# =============================================================================
# PRE-BUILT RESULT FIXTURES (for scorer and reporter tests)
# =============================================================================

@pytest.fixture
def sample_jd_criteria():
    """
    A pre-built JD criteria dict as produced by JobDescriptionParser.parse().
    Used in scorer tests to avoid depending on the JD parser being correct.
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
    Feature dict for a strong candidate, as produced by ResumeFeatureExtractor.
    """
    return {
        "skills":           ["python", "fastapi", "docker", "postgresql", "aws", "git"],
        "min_experience":   8,
        "education_level":  0.8,
        "keywords":         ["python","fastapi", "develop", "backend", "aws", "docker", "git"],
        "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
    }

@pytest.fixture
def weak_candidate_features():
    """
    Feature dict for a weak candidate - no skills, minimal experience.
    """

    return {
        "skills":           [],
        "min_experience":   0,
        "education_level":  0.2,
        "keywords":         ["design","figma", "photoshop"],
        "entities":         {"ORG": [], "GPE": [], "DATE": [], "PERSON": []},
    }