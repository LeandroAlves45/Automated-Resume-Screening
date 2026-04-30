# =============================================================================
# test_pipeline.py - Testes de integração
# =============================================================================
# Cobre os casos IT-01, IT-02, IT-03 e IT-04 da documentação SDLC.
# Estes testes validam a interação entre módulos no fluxo completo:
# parse -> preprocess -> extract -> score -> report.
# =============================================================================

import json
import csv
import pytest

from backend.parser.resume_parser import ResumeParser
from backend.parser.jd_parser import JobDescriptionParser
from backend.nlp.preprocessor import TextPreprocessor
from backend.nlp.extractor import ResumeFeatureExtractor
from backend.scoring.scorer import ResumeScorer
from backend.reports.reporter import ReportGenerator
from backend.api.cli import process_resumes, sort_results, load_job_description


@pytest.fixture
def pipeline(nlp_model, tmp_path):
    """
    Cria uma instância fresca de cada componente do pipeline.
    """

    return {
        "resume_parser": ResumeParser(),
        "jd_parser": JobDescriptionParser(nlp_model),
        "preprocessor": TextPreprocessor(nlp_model),
        "extractor": ResumeFeatureExtractor(nlp_model),
        "scorer": ResumeScorer(),
        "reporter": ReportGenerator(output_dir=str(tmp_path/ "output"))
    }


@pytest.fixture
def populated_folder(tmp_path, sample_jd_text):
    """
    Cria uma pasta temporária com três CVs distintos e uma vaga.
    """

    strong_cv = """
    Alice Senior â€” Lead Python Developer
    Experience at TechCorp (2018 - 2024).
    I have 6 years of experience in backend development.
    Skills: Python, FastAPI, Docker, PostgreSQL, AWS, Git.
    MSc Computer Science 2017.
    """

    average_cv = """
    Bob Mid â€” Python Developer
    Developer at SmallCo (2021 - 2024).
    Skills: Python, Docker, Git.
    Bachelor in Computer Science 2020.
    """

    weak_cv = """
    Carol Junior â€” Marketing Assistant
    Skills: Excel, PowerPoint, Communication.
    High School Certificate 2022.
    I am looking for my first job.
    This is a longer text to pass the minimum length check for testing purposes ok.
    """

    cv_folder = tmp_path / "resumes"
    cv_folder.mkdir()

    (cv_folder / "alice.txt").write_text(strong_cv, encoding="utf-8")
    (cv_folder / "bob.txt").write_text(average_cv, encoding="utf-8")
    (cv_folder / "carol.txt").write_text(weak_cv, encoding="utf-8")

    jd_file = tmp_path / "jd.txt"
    jd_file.write_text(sample_jd_text, encoding="utf-8")

    return {"cv_folder": cv_folder, "jd_file": jd_file, "output": tmp_path / "output"}


class TestFullPipelineRanking:

    def test_it01_strong_candidate_ranks_above_weak(self, pipeline, populated_folder):
        """
        IT-01: candidato forte deve ficar acima do candidato fraco no ranking.
        """
        p= pipeline
        f= populated_folder

        # Processa a descrição da vaga.
        jd_criteria = load_job_description(
            str(f["jd_file"]), p["jd_parser"]
        )
        assert jd_criteria is not None

        # Executa o pipeline completo para todos os CVs.
        scored_results, failed = process_resumes(
            folder_path=str(f["cv_folder"]),
            jd_criteria=jd_criteria,
            resume_parser=p["resume_parser"],
            preprocessor=p["preprocessor"],
            extractor=p["extractor"],
            scorer=p["scorer"],
        )

        # Todos os três CVs devem ser processados.
        assert len(failed) == 0
        assert len(scored_results) == 3

        # Ordena como main.py ordenaria.
        ranked = sort_results(scored_results)

        # Alice deve pontuar acima de Carol.
        scores = {r["name"]: r["total_score"] for r in ranked}
        assert scores["alice"] > scores["carol"]

        # O ranking esperado coloca Alice primeiro e Carol por último.
        assert ranked[0]["name"] == "alice"
        assert ranked[-1]["name"] == "carol"


class TestCSVOutput:

    def test_it02_csv_has_header_plus_one_row_per_candidate(self, pipeline, populated_folder):
        """
        IT-02: o CSV deve ter N+1 linhas: cabeçalho + candidatos.
        """
        p = pipeline
        f = populated_folder

        jd_criteria = load_job_description(str(f["jd_file"]), p["jd_parser"])
        scored_results, _ =process_resumes(
            str(f["cv_folder"]), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )
        ranked = sort_results(scored_results)

        csv_path = p["reporter"].save_csv(ranked)

        # Lê todas as linhas do CSV gerado.
        with open(csv_path, encoding="utf-8-sig") as f_csv:
            reader = csv.reader(f_csv)
            rows = list(reader)

        # rows[0] é o cabeçalho; as restantes linhas são candidatos.
        assert len(rows) == len(ranked) + 1


class TestJSONOutput:

    def test_it03_json_is_parseable_and_has_expected_structure(self, pipeline, populated_folder):
        """
        IT-03: o JSON deve ser válido e conter metadata e candidates.
        """
        p = pipeline
        f = populated_folder

        jd_criteria = load_job_description(str(f["jd_file"]), p["jd_parser"])
        scored_results, _ = process_resumes(
            str(f["cv_folder"]), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )
        ranked = sort_results(scored_results)

        json_path = p["reporter"].save_json(ranked)

        # Lê e interpreta o ficheiro JSON.
        with open(json_path, encoding="utf-8") as f_json:
            data = json.load(f_json)

        # Estrutura de topo esperada.
        assert "metadata" in data
        assert "candidates" in data

        # Metadata principal.
        assert data["metadata"]["total_candidates"] == len(ranked)

        # Cada candidato deve expor os campos necessários.
        for candidate in data["candidates"]:
            assert "rank"                   in candidate
            assert "name"                   in candidate
            assert "total_score"            in candidate
            assert "category"               in candidate
            assert "breakdown"              in candidate
            assert "matched_skills"         in candidate
            assert "missing_skills"
            assert "experience_years_found" in candidate

    def test_json_scores_are_within_valid_range(self, pipeline, populated_folder):
        """
        IT-04: todas as pontuações no JSON ficam entre 0 e 100.
        """
        p = pipeline
        f = populated_folder

        jd_criteria = load_job_description(str(f["jd_file"]), p["jd_parser"])
        scored_results, _ = process_resumes(
            str(f["cv_folder"]), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )
        json_path = p["reporter"].save_json(sort_results(scored_results))

        with open(json_path, encoding="utf-8") as f_json:
            data = json.load(f_json)

        for candidate in data["candidates"]:
            assert 0.0 <= candidate["total_score"] <= 100.0


class TestRobustness:

    def test_it04_invalid_file_does_not_block_valid_cvs(self, pipeline, tmp_path, sample_jd_text, strong_cv_text):
        """
        IT-04: um CV inválido não bloqueia o processamento dos CVs válidos.
        """

        p = pipeline

        cv_folder = tmp_path / "resumes"
        cv_folder.mkdir()

        # CV válido que deve ser pontuado.
        (cv_folder / "valid_candidate.txt").write_text(strong_cv_text, encoding="utf-8")

        # Ficheiro curto demais para passar a validação mínima.
        (cv_folder / "too_short.txt").write_text("Hi.", encoding="utf-8")

        # Extensão não suportada, ignorada silenciosamente.
        (cv_folder / "notes.xyz").write_text("Meeting notes.", encoding="utf-8")

        jd_file = tmp_path / "jd.txt"
        jd_file.write_text(sample_jd_text, encoding="utf-8")

        jd_criteria = load_job_description(str(jd_file), p["jd_parser"])
        scored_results, failed_files = process_resumes(
            str(cv_folder), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )

        # O CV válido deve ser pontuado.
        assert len(scored_results) == 1
        assert scored_results[0]["name"] == "valid_candidate"

        # O ficheiro curto deve aparecer em failed_files.
        assert len(failed_files) == 1
        assert failed_files[0]["name"] == "too_short"


class TestSortOrder:

    def test_sort_by_score_descending(self):
        results = [
            {"name": "a", "total_score": 45.0},
            {"name": "b", "total_score": 80.0},
            {"name": "c", "total_score": 60.0},
        ]

        ranked = sort_results(results)

        assert ranked[0]["total_score"] == 80.0
        assert ranked[1]["total_score"] == 60.0
        assert ranked[2]["total_score"] == 45.0

    def test_sort_alphabetically_on_score_tie(self):
        results = [
            {"name": "zara",  "total_score": 70.0},
            {"name": "alice", "total_score": 70.0},
            {"name": "mike",  "total_score": 70.0},
        ]
        ranked = sort_results(results)

        assert ranked[0]["name"] == "alice"
        assert ranked[1]["name"] == "mike"
        assert ranked[2]["name"] == "zara"
