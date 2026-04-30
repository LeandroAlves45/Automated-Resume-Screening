# =============================================================================
# test_resume_parser.py - Testes unitários do ResumeParser
# =============================================================================
# Cobre os casos UT-01, UT-02 e UT-03 definidos na documentação SDLC.
# Os testes usam ficheiros reais criados em tmp_path.
# =============================================================================

from unittest import result

import pytest
from backend.parser.resume_parser import ResumeParser


@pytest.fixture
def parser():
    return ResumeParser()


class TestParseSingleFile:

    def test_ut01_valid_txt_file_returns_text(self, parser, tmp_path, strong_cv_text):
        """
        UT-01: um TXT válido devolve texto preenchido e error = None.
        """

        # Arrange: cria um ficheiro real na pasta temporária.
        cv_file = tmp_path / "candidate.txt"
        cv_file.write_text(strong_cv_text, encoding="utf-8")

        # Act: processa o ficheiro.
        result = parser.parse(str(cv_file))

        # Assert: parsing bem-sucedido devolve texto e não devolve erro.
        assert result["error"] is None
        assert len(result["text"]) >= 100
        assert result["name"] == "candidate"
        assert result["file"] == str(cv_file)

    def test_ut02_nonexistent_file_returns_error(self, parser, tmp_path):
        """
        UT-02: ficheiro inexistente devolve erro e texto vazio, sem exceção.
        """
        # Arrange: caminho para ficheiro que não existe.
        missing_path = str(tmp_path / "ghost.txt")

        # Act
        result = parser.parse(missing_path)

        # Assert: erro preenchido e texto vazio.
        assert result["error"] is not None
        assert "not found" in result["error"].lower()
        assert result["text"] == ""

    def test_ut03_cv_too_short_returns_error(self, parser, tmp_path, short_cv_text):
        """
        UT-03: CV com menos de 100 caracteres é considerado inválido.
        """
        # Arrange: cria um ficheiro com conteúdo muito curto.
        cv_file = tmp_path / "tiny.txt"
        cv_file.write_text(short_cv_text, encoding="utf-8")

        # Act
        result = parser.parse(str(cv_file))

        # Assert: o erro menciona o tamanho insuficiente.
        assert result["error"] is not None
        assert "too short" in result["error"].lower()
        assert result["text"] == ""

    def test_unsupported_extension_returns_error(self, parser, tmp_path):
        """
        Extensão não suportada devolve erro e não tenta ler o ficheiro.
        """

        xyz_file = tmp_path / "document.xyz"
        xyz_file.write_text("Some content", encoding="utf-8")

        result = parser.parse(str(xyz_file))

        assert result["error"] is not None
        assert "unsupported" in result["error"].lower()

    def test_result_dict_always_has_all_keys(self, parser, tmp_path):
        """
        O resultado deve conter sempre as quatro chaves esperadas.
        """
        # Usa um ficheiro em falta para acionar o caminho de falha.
        result = parser.parse(str(tmp_path / "missing.pdf"))

        # Todas as chaves devem estar presentes.
        assert "name" in result
        assert "file" in result
        assert "text" in result
        assert "error" in result

    def test_txt_encoding_fallback_latin1(self, parser, tmp_path):
        """
        Um TXT em Latin-1 é lido pelo fallback de encoding.
        """
        # Escreve um ficheiro Latin-1 com caracteres acentuados.
        cv_file = tmp_path / "latin_cv.txt"
        content = "JoÃ£o Silva\nSenior Developer\n" + "Python experience." * 10
        cv_file.write_bytes(content.encode("latin-1"))

        result = parser.parse(str(cv_file))

        # O ficheiro deve ser lido sem erro apesar de não estar em UTF-8.
        assert result["error"] is None
        assert "Jo" in result["text"]  # O nome foi extraído.


class TestParseFolder:

    def test_folder_returns_pnly_supported_files(self, parser, tmp_cv_folder):
        """
        parse_folder() devolve resultados só para extensões suportadas.
        """
        results = parser.parse_folder(str(tmp_cv_folder))

        # Apenas ficheiros .txt e .pdf devem ser processados.
        assert len(results) == 2

    def test_folder_nonexistent_returns_empty_list(self, parser, tmp_path):
        """
        Pasta inexistente devolve lista vazia, sem exceção.
        """

        results = parser.parse_folder(str(tmp_path / "no_such_folder"))
        assert results == []

    def test_folder_results_sorted_alphabetically(self, parser, tmp_cv_folder):
        """
        Resultados de parse_folder() vêm ordenados alfabeticamente por nome.
        """
        results = parser.parse_folder(str(tmp_cv_folder))
        names = [r["name"] for r in results]

        # A ordem deve ser alfabética.
        assert names == sorted(names)

    def test_bad_file_does_not_abort_batch(self, parser, tmp_path, strong_cv_text):
        """
        Um ficheiro inválido não impede o processamento dos restantes.
        """

        # Escreve um CV válido.
        (tmp_path / "valid_cando.txt").write_text(strong_cv_text, encoding="utf-8")
        # Escreve um CV inválido por ser demasiado curto.
        (tmp_path / "empty_cv.txt").write_text("Too short", encoding="utf-8")

        results = parser.parse_folder(str(tmp_path))

        # Ambos os ficheiros suportados são tentados.
        assert len(results) == 2

        # Um deve passar e um deve falhar.
        successes = [r for r in results if r["error"] is None]
        failures = [r for r in results if r["error"] is not None]
        assert len(successes) == 1
        assert len(failures) == 1
