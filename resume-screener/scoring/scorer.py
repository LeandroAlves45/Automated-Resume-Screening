# =============================================================================
# scorer.py — Weighted Scoring Engine
# =============================================================================
# Responsibility: Calculate a final weighted score [0, 100] for each candidate
# by combining four independent scoring criteria.
#
# The four criteria and their default weights (from config.py):
#   skills_match      40% — how many required JD skills the CV contains
#   experience_years  25% — candidate years vs JD minimum (linear curve)
#   education         15% — candidate education level vs JD requirement
#   keyword_density   20% — TF-IDF cosine similarity between CV and JD text
#
# Design decisions:
#   - Each criterion is calculated by its own private method (SRP).
#   - All criteria return a float in [0.0, 1.0] before weighting.
#   - The final score is clipped to [0, 100] using numpy (business rule §2.3).
#   - TF-IDF operates on lemmatised token lists, not raw text, so morphological
#     variation ("developed", "developer") is normalised before vectorisation.
#
# SOLID principle: Open/Closed Principle (OCP).
# New scoring criteria can be added by defining a new private method and
# registering it in score() — no existing method needs to change.
# =============================================================================

import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import SCORING_WEIGHTS, THRESHOLDS

logger = logging.getLogger(__name__)

class ResumeScorer:
    """
    Calculates a weighted score for a candidate CV against a Job Description.
 
    Combines four independent criteria into a single score in [0, 100].
    Each criterion is isolated in its own method for testability and clarity.
    """

    # -----------------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------------

    def score(self,resume_features: dict, jd_criteria: dict) -> dict:
        """
        Calculate the final weighted score for a single candidate.
 
        Orchestrates the four criterion calculations, applies weights from
        config.py, clips the result to [0, 100], and classifies the candidate.
 
        Args:
            resume_features (dict): Output of ResumeFeatureExtractor.extract().
                Required keys: matched_skills, experience_years,
                               education_level, keywords.
            jd_criteria (dict): Output of JobDescriptionParser.parse().
                Required keys: skills, min_experience,
                               education_level, keywords, raw_text.
 
        Returns:
            dict: {
                "total_score":            float,      # Final score [0.0, 100.0]
                "breakdown":              dict,        # Per-criterion scores (0-100 each)
                "category":               str,         # Strong / Potential / Weak Match
                "matched_skills":         list[str],   # Skills found in CV
                "missing_skills":         list[str],   # Required skills NOT in CV
                "experience_years_found": int          # Years detected in CV
            }
        """
        # --- Calculate each criterion score individually ---
        # Each method returns a float in [0.0, 1.0].
        # We calculate all four before applying weights so that each one
        # can be logged and inspected in the breakdown independently.

        skills_score = self._score_skills(matched=resume_features.get("matched_skills", []), required=jd_criteria.get("skills", []))

        experience_score = self._score_experience(candidate_years=resume_features.get("experience_years", 0), min_required=jd_criteria.get("min_experience", 0))

        education_score = self._score_education(candidate_level=resume_features.get("education_level", 0.0), required_level=jd_criteria.get("education_level", 0.0))

        keyword_score = self._score_keywords(resume_keywords=resume_features.get("keywords", []), jd_keywords=jd_criteria.get("keywords", []))

        # --- Apply weights and calculate final score ---
        # Each score is multiplied by its weight from config.py.
        # The result is a float in [0.0, 1.0] (since weights sum to 1.0
        # and each score is in [0.0, 1.0]).
        weights = SCORING_WEIGHTS
        weighted_sum = (
            skills_score * weights["skills_match"] +
            experience_score * weights["experience_years"] +
            education_score * weights["education"] +
            keyword_score * weights["keyword_density"]
        )

        # --- Convert to 0-100 scale and clip ---
        # Multiply by 100 to get a human-readable percentage.
        # numpy.clip() enforces the business rule that no score can be
        # below 0 or above 100, regardless of floating point rounding.
        # We round to 1 decimal place for clean output in reports.
        raw_score_100 = weighted_sum * 100
        total_score = float(np.clip(round(raw_score_100, 1), 0.0, 100.0))

        # --- Build the per-criterion breakdown ---
        # Multiply each normalised score by 100 for readability.
        # This breakdown is shown in the detailed report and is essential
        # for a recruiter to understand why a candidate received their score.
        breakdown = {
            "skills_match": round(skills_score * 100, 1),
            "experience_years": round(experience_score * 100, 1),
            "education": round(education_score * 100, 1),
            "keyword_density": round(keyword_score * 100, 1)
        }

        # --- Classify candidate based on thresholds ---
        category = self._classify(total_score)

        # --- Determine missing skills ---
        # Missing = required by JD but not found in CV.
        # This is reported to help the recruiter understand the skills gap.
        all_required = set(jd_criteria.get("skills", []))
        matched_set = set(resume_features.get("matched_skills", []))
        missing_skills = sorted(all_required - matched_set)

        result ={
            "total_score": total_score,
            "breakdown": breakdown,
            "category": category,
            "matched_skills": resume_features.get("matched_skills", []),
            "missing_skills": missing_skills,
            "experience_years_found": resume_features.get("experience_years", 0)
        }

        logger.info(
            f"Score: {total_score} ({category}) | "
            f"Skills: {breakdown['skills_match']} | "
            f"Exp: {breakdown['experience_years']} | "
            f"Edu: {breakdown['education']} | "
            f"KW: {breakdown['keyword_density']} | "
        )
        return result
    
    # -----------------------------------------------------------------------------
    # PRIVATE METHODS FOR EACH CRITERION
    # -----------------------------------------------------------------------------

    def _score_skills(self, matched: list, required: list) -> float:
        """
        Calculate the skills match score.
 
        Formula: matched_skills / total_required_skills
        Result is naturally in [0.0, 1.0] since matched <= required always.
 
        Neutral case: if the JD lists no required skills, we cannot penalise
        or reward the candidate for skills — return 0.5 (neutral) per the
        business rule in SDLC. This prevents an empty JD skills list
        from zeroing out a criterion that should be treated as unknown.
 
        Args:
            matched (list[str]): Skills from the JD that the CV contains.
            required (list[str]): All skills listed in the JD.
 
        Returns:
            float: Score in [0.0, 1.0].
        """
        # Neutral case: ratio of matched to required
        if not required:
            return 0.5
        
        # Standard case: ratio of matched to required
        score = len(matched) / len(required)

        logger.debug(f"Skills: {len(matched)}/{len(required)} = {score:.3f}")

        return score

    def _score_experience(self, candidate_years: int, min_required: int) -> float:
        """
        Calculate the experience score using a linear curve.
 
        The scoring curve works as follows:
          - 0 years when minimum is required       → 0.0
          - Exactly at the minimum                 → 0.5
          - At 2x the minimum (or above)           → 1.0 (capped)
 
        The curve is linear between 0 and 2x minimum:
          score = candidate_years / (2 * min_required)
 
        Why 2x as the cap? Because someone with twice the required experience
        is clearly well above the threshold — awarding them maximum points is
        fair without creating an unbounded advantage for very senior candidates.
 
        Neutral case: if the JD specifies no minimum, return 0.5 (neutral)
        — we cannot evaluate experience when no baseline is defined.
 
        Args:
            candidate_years (int): Years of experience detected in the CV.
            min_required (int): Minimum years required by the JD (0 = not stated).
 
        Returns:
            float: Score in [0.0, 1.0].
        """
        # Neutral case: no minimum specified
        if min_required == 0:
            logger.debug("Experience criterion: no requirement in JD. Returning neutral 0.5.")
            return 0.5
        
        # Linear curve: score = years / (2 * minimum), capped at 1.0.
        # min() enforces the cap — a candidate with 20 years for a 5-year role
        # gets 1.0, not 2.0.
        score = min(candidate_years / (2 * min_required), 1.0)

        logger.debug(
            f"Experience: {candidate_years / (2 * min_required)} = {score:.3f} "
        )
        return score
    
    def _score_education(self, candidate_level: float, required_level: float) -> float:
        """
        Calculate the education score.
 
        Formula: candidate_level / required_level, capped at 1.0.
 
        A candidate with a Master's (0.8) applying for a role requiring
        a Bachelor's (0.6) receives 0.8/0.6 = 1.33 → capped to 1.0.
        A candidate with a Bachelor's (0.6) applying for a Master's role
        (0.8) receives 0.6/0.8 = 0.75 — a partial score, not zero.
 
        Neutral case: if NEITHER the JD nor the CV mentions education,
        return 0.6 (slightly positive neutral) per SDLC §4.2. This avoids
        penalising candidates when education is simply not mentioned.
 
        Args:
            candidate_level (float): Education level from CV [0.0, 1.0].
            required_level (float): Education level required by JD [0.0, 1.0].
 
        Returns:
            float: Score in [0.0, 1.0].
        """
        # Neutral case: neither the CV nor the JD mentions education. 0.6 is used (not 0.5).
        if candidate_level == 0.0 and required_level == 0.0:
            logger.debug("Education criterion: no education info in CV or JD. Returning neutral 0.6.")
            return 0.6
        
        # If the JD requires education but the CV mentions none, score is 0.
        # We cannot give partial credit when the candidate provides no information.
        if required_level == 0.0:
            # JD has no requirement — candidate's education is a bonus, not scored.
            # Return a moderate positive value rather than penalising.
            logger.debug("Education criterion: no JD requirement. Returning candidate level.")
            return min(candidate_level, 1.0)
        
        if candidate_level == 0.0:
            # CV has no education info but JD requires it — partial penalty.
            logger.debug("Education criterion: CV has no education info. Returning 0.0.")
            return 0.0

        # Standard case: ratio, capped at 1.0
        score = min(candidate_level / required_level, 1.0)

        logger.debug(
            f"Education: {candidate_level} / {required_level} = {score:.3f}"
        )
        return score
    
    def _score_keywords(self, resume_keywords: list[str], jd_keywords: list[str]) -> float:
        """
        Calculate semantic similarity between CV and JD using TF-IDF + Cosine Similarity.
 
        How TF-IDF works here:
          1. Both keyword lists are joined into single strings (space-separated).
          2. TfidfVectorizer builds a vocabulary from BOTH documents combined.
          3. Each document is represented as a vector of TF-IDF weights.
          4. Cosine Similarity measures the angle between the two vectors:
               - 1.0 = identical term distributions (perfect match)
               - 0.0 = no shared terms at all (no overlap)
 
        Neutral/fallback case: if either list is empty, TF-IDF cannot be
        computed (no vocabulary). Returns 0.0 per SDLC.
 
        Args:
            resume_keywords (list[str]): Lemmatised tokens from the CV.
            jd_keywords (list[str]): Lemmatised keywords from the JD.
 
        Returns:
            float: Cosine similarity in [0.0, 1.0].
        """
        # If either document has no keywords, TF-IDF has nothing to vectorise.
        # We return 0.0 rather than raising an error — an empty keyword list
        # is a valid (if unfortunate) state, not a bug.
        if not resume_keywords or not jd_keywords:
            logger.debug("Keywords criterion: one or both keyword lists are empty. Returning 0.0.")
            return 0.0

        # Join the token lists into space-separated strings.
        # TfidfVectorizer expects a list of documents (strings), not lists of tokens.
        # ' '.join() is the standard way to convert a token list back to a string
        # for vectorisation — the vectoriser will re-split on whitespace internally.
        resume_text = ' '.join(resume_keywords)
        jd_text = ' '.join(jd_keywords)

        try:
            # fit_transform() performs two operations in one call:
            #   fit()      — builds the vocabulary from BOTH documents
            #   transform() — converts each document into a TF-IDF vector
            # The result is a sparse matrix with shape (2, vocabulary_size).
            # We pass BOTH documents together so the vocabulary (and IDF weights)
            # are computed across the full corpus — this is essential for IDF
            # to correctly downweight terms that appear in both documents.
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

            # cosine_similarity() returns a 2x2 matrix of similarities:
            #   [[sim(cv, cv), sim(cv, jd)],
            #    [sim(jd, cv), sim(jd, jd)]]
            # We want sim(cv, jd), which is at position [0][1].
            # [0][0] and [1][1] are always 1.0 (a document is identical to itself).
            similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
            score = float(similarity_matrix[0][0])  # Extract the single similarity value

            logger.debug(f"Keyword cosine similarity: {score:.4f}")
            return score
        
        except Exception as e:
            # If TF-IDF fails for any unexpected reason (e.g. all tokens are
            # filtered by the vectoriser's internal stopword list), we log the
            # error and return 0.0 rather than crashing the pipeline.
            logger.error(f"TF-IDF computation failed: {e}. Returning 0.0.")
            return 0.0
    
    # ----------------------------------------------------------------------
    # CLASSIFICATION
    # ----------------------------------------------------------------------

    def _classify(self, score:float) -> str:
        """
        Classify a candidate into a match category based on their total score.
 
        Thresholds are read from config.py so they can be adjusted without
        touching this method. The classification follows a simple descending
        threshold check — Strong first, then Potential, then Weak as default.
 
        Args:
            score (float): Final total score in [0.0, 100.0].
 
        Returns:
            str: "Strong Match", "Potential Match", or "Weak Match".
        """

        if score >= THRESHOLDS["strong_match"]:
            return "Strong Match"
        elif score >= THRESHOLDS["potential_match"]:
            return "Potential Match"
        else:
            return "Weak Match"