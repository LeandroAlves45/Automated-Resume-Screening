"""
Motor de pontuação ponderada.

Calcula a pontuação final [0, 100] combinando competências, experiência,
formação e similaridade de palavras-chave.
"""

import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.api.scoring_config import SCORING_WEIGHTS, THRESHOLDS

logger = logging.getLogger(__name__)


class ResumeScorer:
    """
    Calcula a pontuação ponderada de um CV face a uma descrição de vaga.
    """

    def score(self, resume_features: dict, jd_criteria: dict) -> dict:
        """
        Calcula score total, breakdown por critério e categoria final.
        """
        # Calcula cada critério numa escala normalizada [0.0, 1.0].
        skills_score = self._score_skills(
            matched=resume_features.get("matched_skills", []),
            required=jd_criteria.get("skills", []),
        )

        experience_score = self._score_experience(
            candidate_years=resume_features.get("experience_years", 0),
            min_required=jd_criteria.get("min_experience", 0),
        )

        education_score = self._score_education(
            candidate_level=resume_features.get("education_level", 0.0),
            required_level=jd_criteria.get("education_level", 0.0),
        )

        keyword_score = self._score_keywords(
            resume_keywords=resume_features.get("keywords", []),
            jd_keywords=jd_criteria.get("keywords", []),
        )

        # Aplica os pesos configurados e converte a soma ponderada para percentagem.
        weights = SCORING_WEIGHTS
        weighted_sum = (
            skills_score * weights["skills_match"]
            + experience_score * weights["experience_years"]
            + education_score * weights["education"]
            + keyword_score * weights["keyword_density"]
        )

        raw_score_100 = weighted_sum * 100
        total_score = float(np.clip(round(raw_score_100, 1), 0.0, 100.0))

        # Breakdown legível para relatórios e auditoria da decisão.
        breakdown = {
            "skills_match": round(skills_score * 100, 1),
            "experience_years": round(experience_score * 100, 1),
            "education": round(education_score * 100, 1),
            "keyword_density": round(keyword_score * 100, 1),
        }

        category = self._classify(total_score)

        # Competências em falta ajudam a explicar a diferença entre CV e vaga.
        all_required = set(jd_criteria.get("skills", []))
        matched_set = set(resume_features.get("matched_skills", []))
        missing_skills = sorted(all_required - matched_set)

        result = {
            "total_score": total_score,
            "breakdown": breakdown,
            "category": category,
            "matched_skills": resume_features.get("matched_skills", []),
            "missing_skills": missing_skills,
            "experience_years_found": resume_features.get("experience_years", 0),
        }

        logger.info(
            "Score: %s (%s) | Skills: %s | Exp: %s | Edu: %s | KW: %s",
            total_score,
            category,
            breakdown["skills_match"],
            breakdown["experience_years"],
            breakdown["education"],
            breakdown["keyword_density"],
        )
        return result

    def _score_skills(self, matched: list, required: list) -> float:
        """
        Calcula a proporção de competências exigidas que foram encontradas.
        """
        # Sem competências exigidas, o critério fica neutro.
        if not required:
            return 0.5

        score = len(matched) / len(required)

        logger.debug("Skills: %d/%d = %.3f", len(matched), len(required), score)

        return score

    def _score_experience(self, candidate_years: int, min_required: int) -> float:
        """
        Calcula experiência com curva linear até 2x o mínimo pedido.
        """
        # Sem mínimo definido na vaga, não há base para penalizar ou recompensar.
        if min_required == 0:
            logger.debug(
                "Experience criterion: no requirement in JD. Returning neutral 0.5."
            )
            return 0.5

        # Candidato com 2x o mínimo ou mais recebe pontuação máxima.
        score = min(candidate_years / (2 * min_required), 1.0)
        ratio = candidate_years / (2 * min_required)

        logger.debug("Experience: %s = %.3f", ratio, score)
        return score

    def _score_education(self, candidate_level: float, required_level: float) -> float:
        """
        Calcula formação comparando nível do candidato com requisito da vaga.
        """
        # Caso neutro: nem CV nem vaga referem formação.
        if candidate_level == 0.0 and required_level == 0.0:
            logger.debug(
                "Education criterion: no education info in CV or JD. Returning neutral 0.6."
            )
            return 0.6

        if required_level == 0.0:
            # Se a vaga não exige formação, o nível do candidato não deve penalizar.
            logger.debug(
                "Education criterion: no JD requirement. Returning candidate level."
            )
            return min(candidate_level, 1.0)

        if candidate_level == 0.0:
            # A vaga exige formação e o CV não a menciona.
            logger.debug(
                "Education criterion: CV has no education info. Returning 0.0."
            )
            return 0.0

        score = min(candidate_level / required_level, 1.0)

        logger.debug(
            "Education: %s / %s = %.3f",
            candidate_level,
            required_level,
            score,
        )
        return score

    def _score_keywords(self, resume_keywords: list[str], jd_keywords: list[str]) -> float:
        """
        Calcula similaridade semântica simples com TF-IDF e cosseno.
        """
        # Sem palavras-chave em qualquer documento, não há vocabulário para comparar.
        if not resume_keywords or not jd_keywords:
            logger.debug(
                "Keywords criterion: one or both keyword lists are empty. Returning 0.0."
            )
            return 0.0

        resume_text = " ".join(resume_keywords)
        jd_text = " ".join(jd_keywords)

        try:
            # fit_transform cria o vocabulário conjunto e vetoriza os dois documentos.
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

            similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
            score = float(similarity_matrix[0][0])  # Valor único da similaridade CV/JD

            logger.debug("Keyword cosine similarity: %.4f", score)
            return score

        except Exception as e:  # pylint: disable=broad-exception-caught
            # Falhas inesperadas de vetorização não devem quebrar o pipeline.
            logger.error("TF-IDF computation failed: %s. Returning 0.0.", e)
            return 0.0

    def _classify(self, score: float) -> str:
        """
        Classifica o candidato com base nos limiares configurados.
        """

        if score >= THRESHOLDS["strong_match"]:
            return "Strong Match"
        if score >= THRESHOLDS["potential_match"]:
            return "Potential Match"
        return "Weak Match"
