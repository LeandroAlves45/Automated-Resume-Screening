# ==========================================================
# resume_parser.py - Parser de ficheiros de CV
# ==========================================================
# Extrai texto simples de CVs em PDF, DOCX e TXT.
# A classe oferece uma interface única para formatos diferentes e devolve
# sempre a mesma estrutura de resultado, incluindo erros de parsing.
# ==========================================================

import logging
from pathlib import Path
from pdfminer.high_level import extract_text as pdf_extract_text # Parser de PDF
from docx import Document # Parser de DOCX

from backend.api.scoring_config import SUPPORTED_EXTENSIONS, TEXT_CONFIG

logger = logging.getLogger(__name__)


classbackend.api.scoring_config
    """
    Lê CVs e devolve texto extraído para o pipeline de NLP.

    Suporta parsing individual com parse() e parsing em lote com parse_folder().
    Falhas são devolvidas no resultado para não interromper o lote inteiro.
    """

    def parse(self, file_path: str) -> dict:
        """
        Processa um único ficheiro de CV.

        A extensão decide qual método privado será usado. Qualquer erro é
        capturado no dicionário de resultado para manter o processamento robusto.
        """

        path = Path(file_path)

        result = {
            "name": path.stem, # Nome do ficheiro sem extensão
            "file": str(path), # Caminho completo como string
            "text": "", # Texto extraído, ou vazio em caso de erro
            "error": None # Mensagem de erro, quando existir
        }

        # Validação rápida antes de tentar abrir o ficheiro.
        if not path.exists():
            result["error"] = f"File not found: {file_path}"
            logger.warning(result["error"])
            return result

        # Só processa extensões explicitamente suportadas.
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            result["error"] = f"Unsupported file type: {extension}"
            logger.warning(result["error"])
            return result

        try:
            if extension == ".pdf":
                raw_text = self._parse_pdf (str(path))
            elif extension == ".docx":
                raw_text = self._parse_docx(str(path))
            elif extension == ".txt":
                raw_text = self._parse_txt(str(path))

            # Normaliza o texto bruto antes de o enviar para NLP.
            result["text"] = self._clean_text(raw_text)

            # CVs demasiado curtos costumam indicar ficheiros vazios ou PDFs digitalizados.
            min_length = TEXT_CONFIG["min_resume_length"]
            if len(result["text"]) < min_length:
                result["error"] = (
                    f"Extracted text too short: ({len(result['text'])} chars."
                    f"Minimum required is {min_length}. File may be image-only or empty. "
                )
                result["text"] = "" # Evita processar texto inválido.
                logger.warning(f"{path.name}: {result['error']}")
            else:
                logger.info(f"Successfully parsed: {path.name}"
                            f"({len(result['text'])} chars.)")
        except Exception as e:
            # Guarda qualquer erro inesperado de bibliotecas externas.
            result["error"] = f"Parsing failed:{type(e).__name__}: {e}"
            logger.error(f"{path.name}: {result['error']}")

        return result

    def parse_folder(self, folder_path:str) -> list[dict]:
        """
        Processa todos os CVs suportados numa pasta.

        A leitura não é recursiva. Ficheiros não suportados são ignorados para
        evitar ruído; ficheiros suportados com erro entram na lista de resultados.
        """

        folder = Path(folder_path)

        if not folder.is_dir():
            logger.error(f"Folder not found or is not a directory: {folder_path}")
            return []

        results = []

        # Ordenação estável para resultados consistentes entre sistemas operativos.
        for file_path in sorted(folder.iterdir()):

            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logging.debug(f"Skipping unsupported file: {file_path.name}")
                continue

            results.append(self.parse(str(file_path)))

        logger.info(f"Folder scan complete: {len(results)} supported file(s) found."
                    f"In {folder_path}")
        return results

    def _parse_pdf(self, path: str) -> str:
        """
        Extrai texto de um PDF usando pdfminer.six.

        Funciona melhor em PDFs com camada de texto. PDFs digitalizados podem
        devolver texto vazio ou pouco fiável.
        """

        text = pdf_extract_text(path)
        return text if text is not None else ""

    def _parse_docx(self, path:str) -> str:
        """
        Extrai texto de um DOCX usando python-docx.

        Junta os parágrafos com quebras de linha para preservar alguma estrutura.
        Tabelas, cabeçalhos, rodapés e caixas de texto não são cobertos.
        """

        doc = Document(path)

        # Cada parágrafo é lido como texto e separado por newline.
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        return "\n".join(paragraphs)

    def _parse_txt(self, path: str) -> str:
        """
        Lê um ficheiro TXT, tentando UTF-8 e depois Latin-1.
        """

        try:
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                return f.read()

        except UnicodeDecodeError:
            # Fallback para ficheiros antigos ou criados em ambientes Windows.
            logger.warning(f"UTF-8 decoding failed for '{path}'."
                           f"Retrying with Latin-1 encoding.")
            with open(path, "r", encoding="latin-1") as f:
                return f.read()

    def _clean_text(self, text: str) -> str:
        """
        Normaliza o texto extraído, reduzindo espaços e linhas em branco.
        """

        if not text:
            return ""

        lines = text.splitlines()

        cleaned_lines = []
        for line in lines:
            # Reduz sequências de espaços dentro da linha a um único espaço.
            cleaned_line = " ".join(line.split())
            cleaned_lines.append(cleaned_line)

        result = "\n".join(cleaned_lines)

        # Mantém no máximo uma linha em branco entre secções.
        import re
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()
