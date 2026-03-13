# ============================================================================
# jd_parser.py - Job Description Parser
# ============================================================================
# Responsibility: Parse a Job Description text and extract structured criteria
# that the scoring engine will use to evaluate each CV.
#
# Design pattern: Dependency Injection.
# The spaCy model (nlp) is NOT loaded inside this class. It is loaded ONCE
# in main.py and injected via the constructor. This avoids loading a ~15MB
# model multiple times when multiple modules need it — a significant
# performance concern.
#
# SOLID principle: Single Responsibility Principle (SRP).
# This module is only responsible for extracting criteria FROM a Job Description.
# It does not score, rank, or parse CVs — those are other modules' concerns.
# =============================================================================

import re 
import logging
from typing import Any # For type hints with complex types

from config import EDUCATION_LEVELS, TEXT_CONFIG

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------
# SKILLS REFERENCE LIST
# -------------------------------------------------------------------------------
# This is the curated list of technical skills the parser recognises.
# Matching is done by checking if any of these terms appear in the JD text.
#
# Known limitation (L-01 in SDLC): This list is hard-coded here in v1.0.
# The planned improvement for v1.1 is to externalise it to a JSON file
# so recruiters can add domain-specific skills without touching source code.
#
# The list is organised by category for readability, but at runtime it is
# treated as a flat set (converted below) for O(1) lookup performance.
# Using a set instead of a list matters when the reference has hundreds of skills.
# -----------------------------------------------------------------------------

_SKILLS_REFERENCE: list[str] = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "kotlin",
    "swift", "php", "scala", "r", "matlab", "perl",

    # Web frameworks and libraries
    "django", "flask", "fastapi", "react", "angular", "vue", "node.js", "express",
    "spring", "rails", "laravel", "asp.net",

    # Databases
    "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch", "cassandra", 
    "dynamodb", "oracle",

    # Cloud and Infrastructure
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible", "jenkins",
    "github actions", "ci/cd",

    # Data and ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "xgboost",
    "spark", "airflow", "dbt", "tableau", "power bi",

    # Tools and pratices
    "git", "linux", "bash", "rest api", "graphql", "microservices", "agile", "scrum", "kanban",
    "jira", "figma",
]

# Convert to a set for O(1) membership lookup.
# When checking if "python" is a known skill, checking a set is instant
# regardless of size. Checking a list requires scanning every element.
_SKILLS_SET: set[str] = {skill.lower() for skill in _SKILLS_REFERENCE}

class JobDescriptionParser:
    """
    Extracts structured evaluation criteria from a Job Description text.
 
    Uses a combination of spaCy NLP (for lemmatisation and named entity
    recognition) and regular expressions (for pattern matching on
    experience requirements) to produce a structured dict of criteria.
 
    The spaCy model must be passed in at construction time (Dependency
    Injection) — it is not loaded internally.
    """

    def __init__(self, nlp:Any) -> None:
        """
        Initialise the parser with a pre-loaded spaCy language model.
 
        Args:
            nlp: A loaded spaCy Language object (e.g. spacy.load('en_core_web_sm')).
                 Passed in from main.py to avoid loading the model multiple times.
        """
        self.nlp = nlp
        logger.debug("JobDescriptionParser initialised")

    # -------------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------------------------------

    def parse(self, jd_text:str) -> dict:
        """
        Parse a Job Description and extract structured evaluation criteria.
 
        This is the main entry point for this class. It orchestrates calls
        to all private extraction methods and assembles the final result dict.
 
        Args:
            jd_text (str): The raw text content of the Job Description file.
 
        Returns:
            dict: {
                "skills":           list[str],  # Technical skills required by the JD
                "min_experience":   int,         # Minimum years of experience (0 if not stated)
                "education_level":  float,       # Required education level [0.0, 1.0]
                "keywords":         list[str],   # Lemmatised meaningful keywords from the JD
                "raw_text":         str          # Original JD text (needed by scorer for TF-IDF)
            }
        """
        # Guard against empty or whitespace-only input.
        # An empty JD would produce neutral scores across all criteria,
        # which is misleading — better to log a warning and return defaults.
        if not jd_text or not jd_text.strip():
            logger.warning("Job Description text is empty. All criteria will use neutral values.")
            return self.empty_result(jd_text)
        
        # Run the spaCy NLP pipeline on the full JD text.
        # This single call performs tokenisation, POS tagging, dependency
        # parsing, NER, and lemmatisation — all at once. The result is a
        # spaCy Doc object that all extraction methods will read from.
        # We do this ONCE here and pass the doc to each method to avoid
        # running the pipeline multiple times on the same text.
        doc = self._nlp(jd_text)

        # Extract each criterion independently.
        # Each method reads the same doc and the same raw text.
        # This separation means each extraction can be tested in isolation.
        skills = self._extract_skills(jd_text)
        min_experience = self._extract_min_experience(jd_text)
        education_level = self._extract_education_level(jd_text)
        keywords = self._extract_keywords(doc)

        result = {
            "skills": skills,
            "min_experience": min_experience,
            "education_level": education_level,
            "keywords": keywords,
            "raw_text": jd_text,
        }

        logger.info(
            f"JD parsed - Skills: {len(skills)}, "
            f"Min experience: {min_experience} years, "
            f"Education level: {education_level}, "
            f"Keywords: {len(keywords)}"
        )

        return result
    
    # -------------------------------------------------------------------------------
    # PRIVATE EXTRACTION METHODS
    # -------------------------------------------------------------------------------

    def _extract_skills(self, text:str) -> list[str]:
        """
        Identify technical skills mentioned in the JD text.
 
        Matching strategy: check if each skill in the reference list appears
        as a substring in the lowercased JD text. This is intentionally simple
        for v1.0 — a more robust approach (planned for v2.0) would use
        semantic embeddings to match synonyms (e.g. "Postgres" -> "PostgreSQL").
 
        Args:
            text (str): Raw JD text.
 
        Returns:
            list[str]: Skills from the reference list found in the text,
                       in the order they appear in the reference list.
        """
        # Lowercase the entire text once, then check membership.
        # We avoid lowercasing inside the loop to keep the operation O(n)
        # in the number of skills rather than O(n*m) where m is text length.
        text_lower = text.lower()

        found_skills = []
        for skill in _SKILLS_REFERENCE:
            # Check for the skill as a substring. We wrap it with word boundaries
            # implicitly by checking the full token — but for multi-word skills
            # like "github actions" this still works correctly as a substring match.
            if skill in text_lower:
                found_skills.append(skill)

        logger.debug(f"Skills extracted: {found_skills}")
        return found_skills

    def _extract_min_experience(self, text:str) -> int:
        """
        Extract the minimum years of experience required from the JD text.
 
        Uses 4 regular expression patterns to cover the most common ways
        experience requirements are phrased in job descriptions. Returns
        the MAXIMUM value found across all patterns — if a JD says both
        "3+ years" in one section and "minimum 5 years" in another, we
        return 5 (the stricter requirement).
 
        If no experience requirement is found, returns 0 (per business rule
        in section 2.3: the scorer will then apply a neutral score of 0.5).
 
        Args:
            text (str): Raw JD text.
 
        Returns:
            int: Minimum years of experience required, or 0 if not stated.
        """
        text_lower = text.lower()

        # Each pattern targets a different phrasing convention.
        # (\d+) captures one or more digits — this is the number we want.
        # We use raw strings (r"...") so backslashes are treated as regex
        # syntax, not Python escape sequences.
        patterns = [
            r"(\d+)\+?\s*years?\s*of\s*experience",  # "5+ years experience"
            r"(\d+)\*?\s*years?\s*experience",   # "3 years experience"
            r"minimum\s+(\d+)\s*years?",         # "minimum 3 years"
            r"at\s*least\s+(\d+)\s*years?",     # "at least 2 years"
        ]

        found_values: list[int] = []

        for pattern in patterns:
            # re.findall returns all non-overlapping matches as a list of strings.
            # Each match is the captured group (\d+), not the full match.
            matches = re.findall(pattern, text_lower)
            for match in matches:
                value = int(match)
                # Cap at max_experience_years to prevent false positives.
                # Without this cap, a phrase like "joined in 2019" could be
                # misinterpreted as requiring 2019 years of experience if a new pattern is added incorrectly.
                max_years = TEXT_CONFIG["max_experience_years"]
                if value <= max_years:
                    found_values.append(value)

        if not found_values:
            logger.debug("No experience requirement found in JD.")
            return 0
        
        # Return the strictest (highest) requirement found
        result = max(found_values)
        return result
    
    def _extract_education_level(self, text:str) -> float:
        """
        Identify the highest education level required by the JD.
 
        Scans the JD text for known education keywords (from EDUCATION_LEVELS
        in config.py) and returns the numeric value of the highest match found.
 
        If no education requirement is mentioned, returns 0.0. The scorer
        will then apply a neutral value (0.6) per the business rules in
        section 4.2 of the SDLC documentation.
 
        Args:
            text (str): Raw JD text.
 
        Returns:
            float: Normalised education level [0.0, 1.0], or 0.0 if not found.
        """
        text_lower = text.lower()

        found_levels:list[float] = []

        for keyword, level_value in EDUCATION_LEVELS.items():
            if keyword in text_lower:
                found_levels.append(level_value)

        if not found_levels:
            logger.debug("No education requirement found in JD.")
            return 0.0
        
        # Return the highest level found, not the first.
        # A JD that mentions both "bachelor" (0.6) and "master preferred" (0.8)
        # should be treated as requiring a master's level.
        result = max(found_levels)
        logger.debug(f"Education level extracted: {result}")
        return result

    def _extract_keywords(self, doc:Any) -> list[str]:
        """
        Extract lemmatised, meaningful keywords from the JD using spaCy.
 
        Filters out stop words, punctuation, whitespace tokens, and short
        tokens (< 2 characters) that carry no semantic value. The remaining
        tokens are lemmatised (reduced to their base form) and lowercased.
 
        Lemmatisation example: "required", "requiring", "requires" → "require"
        This normalisation ensures that TF-IDF treats these as the same term,
        improving the quality of the semantic similarity score.
 
        Args:
            doc: A spaCy Doc object produced by running nlp(jd_text).
 
        Returns:
            list[str]: Lemmatised keywords, deduplicated and lowercased.
        """
        keywords = []

        for token in doc:
            # Skip stop words — words like "the", "a", "is", "of" that appear
            # everywhere and carry no meaningful signal for our scoring.
            if token.is_stop:
                continue

            # Skip punctuation marks, spaces, and newlines tokens
            if token.is_punct or token.is_space:
                continue

            # Use the lemma (base form) rather than the raw text, lowercased.
            # token.lemma_ returns the lemma as spaCy determines it.
            # For a blank model (used in testing), lemma_ equals the original text.
            lemma = token.lemma_.lower().strip()

            if lemma: # Avoid empty strings after stripping
                keywords.append(lemma)
        
        # Deduplicate while preserving order using dict.fromkeys().
        # A set would deduplicate but destroy order; dict.fromkeys() preserves
        # insertion order (guaranteed in Python 3.7+) while removing duplicates.
        unique_keywords = list(dict.fromkeys(keywords))

        logger.debug(f"Keywords extracted: {len(unique_keywords)} unique terms.")
        return unique_keywords
    
    # -------------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------------

    def _empty_result(self, raw_text:str) -> dict:
        """
        Return a result dict with safe default values for an empty JD.
 
        Used when the JD text is empty or whitespace-only. All criteria
        will default to their neutral values during scoring.
 
        Args:
            raw_text (str): The original (empty) JD text.
 
        Returns:
            dict: Result dict with empty/zero defaults for all fields.
        """
        return {
            "skills": [],
            "min_experience": 0,
            "education_level": 0.0,
            "keywords": [],
            "raw_text": raw_text or "",
        }





