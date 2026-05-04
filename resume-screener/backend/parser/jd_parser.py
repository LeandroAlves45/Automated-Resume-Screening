"""
Parser de descrições de vaga.

Extrai critérios estruturados de uma Job Description para o motor de scoring.
O modelo spaCy é injetado no construtor para ser carregado apenas uma vez
durante a execução do pipeline.
"""

import re
import logging
from typing import Any  # Tipos usados em objetos externos como spaCy Doc/Language

from backend.api.scoring_config import EDUCATION_LEVELS, TEXT_CONFIG

logger = logging.getLogger(__name__)

# Lista curada de competências técnicas reconhecidas pelo parser.
# Em runtime é convertida para set para pesquisas rápidas.
_SKILLS_REFERENCE: list[str] = [
    # Linguagens de programação
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c#",
    "go",
    "rust",
    "kotlin",
    "backend.api.scoring_configala",
    "r",
    "matlab",
    "perl",
    # Frameworks e bibliotecas web
    "django",
    "flask",
    "fastapi",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "spring",
    "rails",
    "laravel",
    "asp.net",
    # Bases de dados
    "postgresql",
    "mysql",
    "sqlite",
    "mongodb",
    "redis",
    "elasticsearch",
    "cassandra",
    "dynamodb",
    "oracle",
    # Cloud e infraestrutura
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "ansible",
    "jenkins",
    "github actions",
    "ci/cd",
    # Dados e machine learning
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "keras",
    "xgboost",
    "spark",
    "airflow",
    "dbt",
    "tableau",
    "power bi",
    # Ferramentas e práticas
    "git",
    "linux",
    "bash",
    "rest api",
    "graphql",
    "microservices",
    "agile",
    "scrum",
    "kanban",
    "jira",
    "figma",
]

_SKILLS_SET: set[str] = {skill.lower() for skill in _SKILLS_REFERENCE}


class JobDescriptionParser:
    """
    Extrai critérios de avaliação a partir do texto de uma vaga.

    Combina spaCy para lematização com expressões regulares para requisitos
    explícitos como anos mínimos de experiência.
    """

    def __init__(self, nlp: Any) -> None:
        """
        Inicializa o parser com um modelo spaCy já carregado.
        """
        self._nlp = nlp
        logger.debug("JobDescriptionParser initialised")

    def parse(self, jd_text: str) -> dict:
        """
        Analisa uma descrição de vaga e devolve critérios estruturados.
        """
        # Uma vaga vazia geraria pontuações neutras enganadoras.
        if not jd_text or not jd_text.strip():
            logger.warning(
                "Job Description text is empty. All criteria will use neutral values."
            )
            return self._empty_result(jd_text)

        # O pipeline spaCy corre uma única vez e o Doc é reutilizado nas extrações.
        doc = self._nlp(jd_text)

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
            "JD parsed — skills: %d, min experience: %d years, education level: %s, keywords: %d",
            len(skills),
            min_experience,
            education_level,
            len(keywords),
        )

        return result

    def _extract_skills(self, text: str) -> list[str]:
        """
        Identifica competências técnicas mencionadas na vaga.

        A correspondência é simples e case-insensitive: cada competência da
        lista de referência é procurada no texto em minúsculas.
        """
        text_lower = text.lower()

        found_skills = []
        for skill in _SKILLS_REFERENCE:
            # Competências muito curtas exigem fronteiras de palavra para evitar falsos positivos.
            if len(skill) < 2:
                if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                    found_skills.append(skill)
            else:
                if skill in text_lower:
                    found_skills.append(skill)

        logger.debug("Skills extracted: %s", found_skills)
        return found_skills

    def _extract_min_experience(self, text: str) -> int:
        """
        Extrai o mínimo de anos de experiência exigido pela vaga.

        Vários padrões cobrem formulações comuns. Quando há mais de um valor,
        é usado o maior por ser o requisito mais restritivo.
        """
        text_lower = text.lower()

        patterns = [
            r"(\d+)\+?\s*years?\s*of\s*experience",  # "5+ years experience"
            r"(\d+)\*?\s*years?\s*experience",  # "3 years experience"
            r"minimum\s+(\d+)\s*years?",  # "minimum 3 years"
            r"at\s*least\s+(\d+)\s*years?",  # "at least 2 years"
        ]

        found_values: list[int] = []

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                value = int(match)
                # Limite de segurança para evitar falsos positivos absurdos.
                max_years = TEXT_CONFIG["max_experience_years"]
                if value <= max_years:
                    found_values.append(value)

        if not found_values:
            logger.debug("No experience requirement found in JD.")
            return 0

        return max(found_values)

    def _extract_education_level(self, text: str) -> float:
        """
        Identifica o nível de formação mais alto pedido na vaga.
        """
        text_lower = text.lower()

        found_levels: list[float] = []

        for keyword, level_value in EDUCATION_LEVELS.items():
            if keyword in text_lower:
                found_levels.append(level_value)

        if not found_levels:
            logger.debug("No education requirement found in JD.")
            return 0.0

        # Usa o nível mais alto mencionado, não o primeiro encontrado.
        result = max(found_levels)
        logger.debug("Education level extracted: %s", result)
        return result

    def _extract_keywords(self, doc: Any) -> list[str]:
        """
        Extrai palavras-chave lematizadas da vaga para comparação TF-IDF.
        """
        keywords = []

        for token in doc:
            if token.is_stop:
                continue
            if token.is_punct or token.is_space:
                continue

            lemma = token.lemma_.lower().strip()

            if lemma:  # Evita strings vazias após strip().
                keywords.append(lemma)

        # Remove duplicados preservando a ordem de inserção.
        unique_keywords = list(dict.fromkeys(keywords))

        logger.debug("Keywords extracted: %d unique terms.", len(unique_keywords))
        return unique_keywords

    def _empty_result(self, raw_text: str) -> dict:
        """
        Devolve valores seguros quando a vaga está vazia.
        """
        return {
            "skills": [],
            "min_experience": 0,
            "education_level": 0.0,
            "keywords": [],
            "raw_text": raw_text or "",
        }
