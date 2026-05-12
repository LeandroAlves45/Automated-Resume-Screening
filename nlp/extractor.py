# =============================================================================
# extractor.py — CV Feature Extractor
# =============================================================================
# Responsibility: Extract quantifiable features from a preprocessed CV text.
# These features feed directly into the scoring engine (scorer.py).
#
# Features extracted:
#   - matched_skills    : JD skills found in the CV text
#   - experience_years  : Years of professional experience
#   - education_level   : Highest education level detected [0.0, 1.0]
#   - keywords          : Lemmatised tokens for TF-IDF scoring
#   - entities          : Named entities detected by spaCy NER
#
# Design pattern: Dependency Injection.
# The spaCy model is injected via the constructor — same instance shared
# across jd_parser.py, preprocessor.py, and this module.
#
# Key technical decision — Two-strategy experience extraction:
# No single regex pattern covers all ways candidates describe their experience.
# Some write "5 years of experience"; others list jobs with date ranges and
# never mention total years. Using both strategies and taking the maximum
# gives the most accurate result for the widest range of CV styles.
# =============================================================================

import re
import logging
from datetime import datetime
from typing import Any

from config import EDUCATION_LEVELS, TEXT_CONFIG

logger = logging.getLogger(__name__)

class ResumeFeatureExtractor:
    """
    Extracts structured, quantifiable features from preprocessed CV text.
 
    All extracted features are normalised or structured so that scorer.py
    can consume them directly without any further interpretation.
    """

    def __init__(self, nlp: Any) -> None:
        """
        Initializes the feature extractor with a spaCy NLP model.

        Args:
            nlp: A loaded spaCy language object, injected from main.py.
        """
        self._nlp = nlp
        logger.debug("ResumeFeatureExtractor initialised.")

    # ----------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ----------------------------------------------------------------------------

    def extract(self, resume_text: str, jd_skills: list[str]) -> dict:
        """
        Run the full feature extraction pipeline on a preprocessed CV text.
 
        Each feature is extracted independently by a dedicated private method.
        This separation means each extraction can fail, be tested, or be
        improved without affecting the others.
 
        Args:
            resume_text (str): Clean CV text from TextPreprocessor.process().
            jd_skills (list[str]): Skills extracted from the JobDescriptionParser.
                                   Used to determine which CV skills are relevant.
 
        Returns:
            dict: {
                "matched_skills":   list[str],   # JD skills found in this CV
                "experience_years": int,          # Total years of experience detected
                "education_level":  float,        # Highest education level [0.0, 1.0]
                "keywords":         list[str],    # Lemmatised tokens for TF-IDF
                "entities":         dict          # NER entities: {ORG, GPE, DATE, PERSON}
            }
        """

        if not resume_text or not resume_text.strip():
            logger.warning("Empty CV text passed to extractor. Returning zero features.")
            return self._empty_result()
        
        # Run the spaCy pipeline once on the full text.
        # The resulting Doc object is shared across all extraction methods
        # that need it (keywords and entities), avoiding redundant computation.
        doc = self._nlp(resume_text)

        matched_skills = self._extract_matched_skills(resume_text, jd_skills)
        experience_years = self._extract_experience(resume_text)
        education_level = self._extract_education(resume_text)
        keywords = self._extract_keywords(doc)
        entities = self._extract_entities(doc)

        result = {
            "matched_skills": matched_skills,
            "experience_years": experience_years,
            "education_level": education_level,
            "keywords": keywords,
            "entities": entities
        }

        logger.info(
            f"Features extracted —"
            f"Skills: {len(matched_skills)}/{len(jd_skills)}, "
            f"Experience: {experience_years} years, "
            f"Education: {education_level}"
        )
        return result
    
    # ----------------------------------------------------------------------------
    # PRIVATE EXTRACTION METHODS
    # ----------------------------------------------------------------------------

    def _extract_matched_skills(self, text: str, jd_skills: list[str]) -> list[str]:
        """
        Identify which JD-required skills are present in the CV text.
 
        Matching is done as a case-insensitive substring search. We search
        the CV for each skill that the JD listed as required — not the
        reverse. A skill only counts if the JD required it.
 
        Args:
            text (str): Clean CV text.
            jd_skills (list[str]): Skills extracted from the JD.
 
        Returns:
            list[str]: Subset of jd_skills found in the CV text.
        """
        # Lowercase the entire text once outside the loop.
        # Doing text.lower() inside the loop would repeat the same operation
        # for every skill — O(n*m) instead of O(n+m).
        text_lower = text.lower()
        matched = []

        for skill in jd_skills:
            if skill.lower() in text_lower:
                matched.append(skill)

        logger.debug(f"Skills matched: {matched}")
        return matched
    
    def _extract_experience(self, text:str) -> int:
        """
        Detect the candidate's years of professional experience.
 
        Implements the two-strategy approach described in SDLC:
          Strategy 1 — Explicit phrases: searches for patterns like
                        "5 years of experience", "over 3 years", etc.
          Strategy 2 — Date range calculation: finds employment date
                        ranges like "2019 - 2023" or "Jan 2020 - Present"
                        and sums their total duration.
 
        Returns the MAXIMUM of both strategies. This is the most
        candidate-favourable interpretation — if either method finds
        more experience, the candidate gets credit for it.
 
        Args:
            text (str): Clean CV text.
 
        Returns:
            int: Detected years of experience, or 0 if none found.
        """
        years_explicit = self._strategy_explicit_phrases(text)
        years_daterange = self._strategy_date_ranges(text)

        result = max(years_explicit, years_daterange)

        # Hard cap from config - prevents regex false positives from producing
        # absurd values (e.g. accidentally capturing "2019" as "2019 years of experience").
        max_years = TEXT_CONFIG.get("max_experience_years")
        result = min(result, max_years)

        logger.debug(
            f"Experinece — explicit: {years_explicit} years, "
            f"Date ranges: {years_daterange} years, final: {result} years"
        ) 
        return result
    
    def _strategy_explicit_phrases(self, text: str) -> int:
        """
        Strategy 1: Extract experience from explicit natural language phrases.
 
        Targets sentences where candidates directly state their experience,
        e.g. "I have 5 years of experience in backend development".
 
        Args:
            text (str): Clean CV text.
 
        Returns:
            int: Maximum years found via this strategy, or 0 if none found.
        """
        text_lower = text.lower()

        patterns = [
            r"(\d+)\+?\s*years?\s*of\s+(?:professional\s+)?experience", # "5 years of experience"
            r"(\d+)\+?\s*years?\s+experience", # "3 years experience"
            r"over\s+(\d+)\+?\s*years?", # "over 3 years"
            r"more\s+than\s+(\d+)\s*years?", # "more than 4 years"
        ]

        found_values: list[int] = []

        for pattern in patterns:
            # re.finditer() yields match objects one at a time — more memory
            # efficient than re.findall() for long texts, and gives us access
            # to named groups if we need them in future pattern iterations.
            for match in re.finditer(pattern, text_lower):
                value = int(match.group(1))
                if value <= TEXT_CONFIG["max_experience_years"]:
                    found_values.append(value)

        return max(found_values) if found_values else 0
    
    def _strategy_date_ranges(self, text: str) -> int:
        """
        Strategy 2: Calculate total experience from employment date ranges.
 
        Scans the CV for date intervals (e.g. "2019 - 2023", "Jan 2020 -
        Present") and sums the total duration across all positions found.
        This works for candidates who list their employment history without
        ever explicitly stating how many years of experience they have.
 
        "Present", "Current", and "Now" are replaced by the current year
        before pattern matching, as specified in SDLC.
 
        Args:
            text (str): Clean CV text.
 
        Returns:
            int: Total years derived from all date ranges found, or 0.
        """
        current_year = datetime.now().year
        text_lower = text.lower()

        # Normalise "present", "current", "now" to the current year STRING
        # before running the date range regex. This lets the same pattern
        # match both "2019 - 2023" and "2019 - Present" without branching.
        text_normalised = re.sub(
            r"\b(present|current|now)\b",
            str(current_year),
            text_lower
        )

        # Matches: "2019 - 2023" | "2019-2023" | "jan 2020 - dec 2022"
        #
        # Pattern breakdown:
        #   (?:\w+\s+)?  — optional month name ("Jan "), non-capturing group
        #   (\d{4})      — capturing group: 4-digit year
        #   \s*[-\u2013]\s*  — hyphen or en-dash separator with optional spaces
        #   (?:\w+\s+)?  — optional month name for end date
        #   (\d{4})      — capturing group: 4-digit end year
        pattern = r"(?:\w+\s+)?(\d{4})\s*[-\u2013]\s*(?:\w+\s+)?(\d{4})"

        total_years = 0

        for match in re.finditer(pattern, text_normalised):
            start_year = int(match.group(1))
            end_year = int(match.group(2))

            # Sanity checks to filter out non-date ranges:
            # - Start must be a plausible career start year (>= 1970)
            # - End must not be in the future
            # - Start must be before end (eliminates reversed number ranges
            #   like version "3.11-3.9" or salary ranges "40000-50000")
            if (start_year >= 1970 and end_year <= current_year and start_year < end_year):
                duration = end_year - start_year
                total_years += duration
                logger.debug(f"Date range: {start_year} - {end_year} => {duration} years")

        return total_years
    
    def _extract_education(self, text: str) -> float:
        """
        Identify the highest education level mentioned in the CV.
 
        Uses EDUCATION_LEVELS from config.py — the same mapping used by
        jd_parser to extract the JD's required level. This ensures the
        candidate and JD education levels are on the same numeric scale,
        making the comparison in scorer.py meaningful.
 
        Args:
            text (str): Clean CV text.
 
        Returns:
            float: Highest education level found [0.0, 1.0].
                   Returns 0.0 if no keywords are detected.
        """
        text_lower = text.lower()
        found_levels: list[float] = []

        for keyword, level_value in EDUCATION_LEVELS.items():
            if keyword in text_lower:
                found_levels.append(level_value)

        if not found_levels:
            logger.debug("No education keywords found in CV.")
            return 0.0
        
        # Return the maximum level found, if have multiple keywords (e.g. "Bachelor's and Master's")
        # holds the most candidate-favourable interpretation.
        return max(found_levels)
    
    def _extract_keywords(self, doc: Any) -> list[str]:
        """
        Extract lemmatised keyword tokens from the CV for TF-IDF scoring.
 
        Filters stop words, punctuation, numeric tokens, and short tokens.
        Duplicates are preserved — TF-IDF needs term frequency information.
 
        Args:
            doc: A spaCy Doc object produced by nlp(resume_text).
 
        Returns:
            list[str]: Filtered, lemmatised, lowercased tokens with duplicates.
        """
        tokens = []

        for token in doc:
            if token.is_stop:
                continue
            if token.is_punct or token.is_space:
                continue
            if token.like_num:
                continue
            if len(token.text) < 2:
                continue

            lemma = token.lemma_.lower().strip()
            if lemma:
                tokens.append(lemma)

        return tokens
    
    def _extract_entities(self, doc: Any) -> dict:
        """
        Extract named entities from the CV using spaCy's NER component.
 
        Collects entities in four categories relevant to CV analysis:
          ORG    — Companies and universities the candidate worked at/attended
          GPE    — Countries and cities (location context)
          DATE   — Date expressions found in the CV
          PERSON — Person names (useful for deduplication in future versions)
 
        Note: NER requires the full en_core_web_sm model. A blank spaCy
        model (used in testing) will return empty entity lists.
 
        Args:
            doc: A spaCy Doc object.
 
        Returns:
            dict: {"ORG": [...], "GPE": [...], "DATE": [...], "PERSON": [...]}
        """
        entities: dict[str, list[str]] = {
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "PERSON": []
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entity_text = ent.text.strip()
                if entity_text:
                    entities[ent.label_].append(entity_text)

        logger.debug(
            f"Entities — ORG: {len(entities['ORG'])}, "
            f"GPE: {len(entities['GPE'])}, "
            f"DATE: {len(entities['DATE'])}," 
        )

        return entities
    
    # ----------------------------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------------------------

    def _empty_result(self) -> dict:
        """
        Return a zero-value result dict for an empty or invalid CV.
 
        Allows the pipeline to continue gracefully when a CV produces
        no extractable text — the candidate receives a score of 0.
 
        Returns:
            dict: All features at their zero/empty defaults.
        """
        return {
            "matched_skills": [],
            "experience_years": 0,
            "education_level": 0.0,
            "keywords": [],
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []}
        }