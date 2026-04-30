# =============================================================================
# extractor.py - Extrator de features de CV
# =============================================================================
# Extrai informação quantificável do texto pré-processado: competências,
# experiência, formação, palavras-chave e entidades nomeadas.
# =============================================================================

import re
import logging
from datetime import datetime
from typing import Any

from backend.api.scoring_config import EDUCATION_LEVELS, TEXT_CONFIG

logger = logging.getLogger(__name__)


class ResumeFeatureExtractor:
    """
    Extrai features estruturadas de um CV já pré-processado.

    As features são devolvidas num formato diretamente consumido pelo scorer.
    """

    def __init__(self, nlp: Any) -> None:
        """
        Inicializa o extrator com um modelo spaCy já carregado.
        """
        self._nlp = nlp
        logger.info("ResumeFeatureExtractor initialised.")

    def extract(self, resume_text: str, jd_skills: list[str]) -> dict:
        """
        Executa a extração completa de features do CV.
        """

        if not resume_text or not resume_text.strip():
            logger.warning("Empty CV text passed to extractor. Returning zero features.")
            return self._empty_result()

        # Corre o spaCy uma vez e reutiliza o Doc nas extrações que precisam dele.
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
            f"Features extracted â€”"
            f"Skills: {len(matched_skills)}/{len(jd_skills)}, "
            f"Experience: {experience_years} years, "
            f"Education: {education_level}"
        )
        return result

    def _extract_matched_skills(self, text: str, jd_skills: list[str]) -> list[str]:
        """
        Identifica quais competências exigidas pela vaga aparecem no CV.
        """
        text_lower = text.lower()
        matched = []

        for skill in jd_skills:
            if skill.lower() in text_lower:
                matched.append(skill)

        logger.debug(f"Skills matched: {matched}")
        return matched

    def _extract_experience(self, text:str) -> int:
        """
        Deteta anos de experiência profissional do candidato.

        Usa duas estratégias: frases explícitas e intervalos de datas. O maior
        valor encontrado é usado para favorecer a interpretação mais completa.
        """
        years_explicit = self._strategy_explicit_phrases(text)
        years_daterange = self._strategy_date_ranges(text)

        result = max(years_explicit, years_daterange)

        # Limite de segurança contra falsos positivos de regex.
        max_years = TEXT_CONFIG.get("max_experience_years")
        result = min(result, max_years)

        logger.debug(
            f"Experinece â€” explicit: {years_explicit} years, "
            f"Date ranges: {years_daterange} years, final: {result} years"
        )
        return result

    def _strategy_explicit_phrases(self, text: str) -> int:
        """
        Extrai experiência declarada em frases como "5 years of experience".
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
            # finditer evita criar listas grandes e mantém acesso ao match completo.
            for match in re.finditer(pattern, text_lower):
                value = int(match.group(1))
                if value <= TEXT_CONFIG["max_experience_years"]:
                    found_values.append(value)

        return max(found_values) if found_values else 0

    def _strategy_date_ranges(self, text: str) -> int:
        """
        Calcula experiência a partir de intervalos de datas no CV.
        """
        current_year = datetime.now().year
        text_lower = text.lower()

        # Normaliza termos de emprego atual para o ano corrente.
        text_normalised = re.sub(
            r"\b(present|current|now)\b",
            str(current_year),
            text_lower
        )

        # Captura intervalos como "2019 - 2023" ou "Jan 2020 - Present".
        pattern = r"(?:\w+\s+)?(\d{4})\s*[-\u2013]\s*(?:\w+\s+)?(\d{4})"

        total_years = 0

        for match in re.finditer(pattern, text_normalised):
            start_year = int(match.group(1))
            end_year = int(match.group(2))

            # Filtra intervalos que não parecem datas de carreira.
            if (start_year >= 1970 and end_year <= current_year and start_year < end_year):
                duration = end_year - start_year
                total_years += duration
                logger.debug(f"Date range: {start_year} - {end_year} => {duration} years")

        return total_years

    def _extract_education(self, text: str) -> float:
        """
        Identifica o nível de formação mais alto mencionado no CV.
        """
        text_lower = text.lower()
        found_levels: list[float] = []

        for keyword, level_value in EDUCATION_LEVELS.items():
            if keyword in text_lower:
                found_levels.append(level_value)

        if not found_levels:
            logger.debug("No education keywords found in CV.")
            return 0.0

        # Usa o nível mais alto encontrado para favorecer o candidato.
        return max(found_levels)

    def _extract_keywords(self, doc: Any) -> list[str]:
        """
        Extrai tokens lematizados do CV para a pontuação TF-IDF.
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
        Extrai entidades nomeadas relevantes para análise de CVs.

        Categorias mantidas: ORG, GPE, DATE e PERSON.
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
            f"Entities â€” ORG: {len(entities['ORG'])}, "
            f"GPE: {len(entities['GPE'])}, "
            f"DATE: {len(entities['DATE'])},"
        )

        return entities

    def _empty_result(self) -> dict:
        """
        Devolve features vazias para CVs sem texto utilizável.
        """
        return {
            "matched_skills": [],
            "experience_years": 0,
            "education_level": 0.0,
            "keywords": [],
            "entities": {"ORG": [], "GPE": [], "DATE": [], "PERSON": []}
        }
