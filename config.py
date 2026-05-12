# ==================================================================
# config.py - Central Configuration Module
# ==================================================================

# This file is the inly place where configurable values should be defined.
# No module in this project should have hardcoded values. Instead, they should import this 
# ==================================================================

# ------------------------------------------------------------------
# SCORING WEIGHTS
# ------------------------------------------------------------------

# Defines the relative importance of each evaluation metric in the overall scoring system.
# These values must sum to exactly 1.0 to ensure a balanced scoring system.

# How to read this:
#  -skills_match is the most important criterion (40%)
#  -experience_years is the second most important (25%)
#  -keyword_density is the third most important (20%)
#  -education is the least important (15%)

SCORING_WEIGHTS: dict[str, float] = {
    "skills_match": 0.4,        # % of required JD skills present in the resume
    "experience_years": 0.25,   # Total years of relevant experience
    "keyword_density": 0.2,     # TF-IDF score of relevant keywords in the resume
    "education": 0.15           # candidate education level vs required level
}

# ------------------------------------------------------------------
# EDUCATION LEVELS
# ------------------------------------------------------------------
# Maps education-related keywords (as they appear in CVs and JDs) to a 
# normalized numeric value between 0.0 and 1.0.
#
# The values represent a relative hierarchy:
#  0.2 = high school (lowest recognition)
#  0.4 = associate / bachelor not yet completed
#  0.6 = bachelor degree 
#  0.8 = master degree
#  1.0 = doctorate (highest recognition)
#
# Keywords are lowercase so matching should always be done after .lower()
# is applied to the extract text.
# -------------------------------------------------------------------
EDUCATION_LEVELS: dict[str, float] = {
    # High School equivalents
    "high school": 0.2,
    "secondary school": 0.2,
    "gcse": 0.2,
    "a-level": 0.2,

    # Sub-degree / associate level
    "associate": 0.4,
    "foundation degree": 0.4,
    "hnd": 0.4,
    "hnc": 0.4,

    # Bachelor level
    "bachelor": 0.6,
    "bsc": 0.6,
    "ba": 0.6,
    "b.sc": 0.6,
    "b.a": 0.6,
    "undergraduate": 0.6,
    "licenciatura": 0.6,

    # Master level
    "master": 0.8,
    "msc": 0.8,
    "mba": 0.8,
    "m.sc": 0.8,
    "postgraduate": 0.8,
    "mestrado": 0.8,

    # Doctorate level
    "phd": 1.0,
    "doctorate": 1.0,
    "ph.d.": 1.0,
    "doctoral": 1.0,
    "doutoramento": 1.0
}


# ------------------------------------------------------------------
# CLASSIFICATION THRESHOLDS
# ------------------------------------------------------------------
# Defines the score boundaries that determine a candidate's category.
# Scores are in the range [0, 100] after multiplying the weighted sum by 100.
#
# Categories:
#   - "Strong Match": >=75 : Strongly aligned profile - advance to interview    
#   - "Potencial Match": >= 50 : Partial relevance - review manually
#   - "Weak Match": < 50 : Not aligned with this role - reject
THRESHOLDS: dict[str, int] = {
    "strong_match": 75,
    "potential_match": 50,
    #anything below potential_match is a weak match
}

# ------------------------------------------------------------------
# TEXT CONFIGURATION
# ------------------------------------------------------------------
#Parameters that control how text is handled across the pipeline
#
#  min_resume_length : A CV with fewer characters than this after extraction is considered invalid and excluded from ranking.
#
#  max_experience_years : Used as a safety cap in the experience extractor to prevent regex false positives (e.g years like  "2019" being interpreted as 2019 years of experience).
#
#  spacy_model: The name of the spacy language model to load. Keeping this here means switching to a larger model.
# -------------------------------------------------------------------
TEXT_CONFIG: dict = {
    "min_resume_length": 100,  # characters - below this, the CV is invalid
    "max_experience_years": 50,  # Maximum years of experience to consider valid
    "spacy_model": "en_core_web_sm"  # Spacy model to use for NLP tasks
}

# -------------------------------------------------------------------
# SUPPORTED FILE EXTENSIONS
# -------------------------------------------------------------------
# The list of file extensions that the resume parser will attempt to process.
# Files with any other extensions will be skipped and logged as unsupported.
# --------------------------------------------------------------------
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt"]