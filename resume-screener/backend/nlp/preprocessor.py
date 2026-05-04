"""
Pré-processador de texto.

Limpa, normaliza e tokeniza texto bruto de CVs antes da extração de features.
Atua apenas como transformação: não interpreta, pontua nem classifica.
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Prepara texto extraído de CVs para o pipeline de NLP.

    Devolve duas representações: texto limpo para regex e tokens lematizados
    para cálculo TF-IDF.
    """

    def __init__(self, nlp: Any) -> None:
        """
        Inicializa o pré-processador com um modelo spaCy já carregado.
        """
        self._nlp = nlp
        logger.debug("TextPreprocessor initialised.")

    def process(self, raw_text: str) -> dict:
        """
        Executa a limpeza e tokenização do texto bruto.
        """
        # Evita gastar processamento com texto vazio.
        if not raw_text or not raw_text.strip():
            logger.warning("Empty text passed to preprocessor. Returning empty results.")
            return {"clean_text": "", "tokens": []}

        clean_text = self._clean_text(raw_text)

        # O spaCy corre sobre texto já normalizado para reduzir ruído.
        doc = self._nlp(clean_text)
        tokens = self._tokenise(doc)

        logger.debug(
            "Preprocessing complete — clean text: %d chars, tokens: %d",
            len(clean_text),
            len(tokens),
        )

        return {
            "clean_text": clean_text,
            "tokens": tokens
        }

    def _clean_text(self, text: str) -> str:
        """
        Normaliza texto bruto removendo caracteres de controlo e ruído comum.
        """
        # Remove caracteres de controlo, preservando newline e tab.
        text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)

        # Converte pontuação Unicode comum em equivalentes ASCII.
        text = text.replace("\u2018", "'").replace("\u2019", "'")  # Aspas simples

        text = text.replace("\u201c", '"').replace("\u201d", '"')  # Aspas duplas

        text = text.replace("\u2013", "-").replace("\u2014", "-")  # Travessões

        text = text.replace("\u2022", "-").replace("\u00b7", "-")  # Marcadores

        # Substitui emails e URLs por placeholders para reduzir ruído nos tokens.
        text = re.sub(r"\S+@\S+\.\S+", " EMAIL ", text)
        text = re.sub(r"https?://\S+", " URL ", text)
        text = re.sub(r"www\.\S+", " URL ", text)

        # Normaliza espaços e linhas em branco.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _tokenise(self, doc: Any) -> list[str]:
        """
        Converte um Doc spaCy numa lista de tokens úteis e lematizados.

        Remove stop words, pontuação, espaços, números puros e tokens curtos.
        Mantém duplicados porque TF-IDF depende da frequência dos termos.
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

            if not lemma:
                continue

            tokens.append(lemma)

        return tokens
