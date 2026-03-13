# =============================================================================
# test_pipeline.py — Integration Tests
# =============================================================================
# Covers test cases IT-01, IT-02, IT-03, IT-04 from SDLC.
#
# Integration tests differ from unit tests in one key way:
# they test the interaction BETWEEN modules, not each module in isolation.
# A unit test checks that extractor.extract() works correctly.
# An integration test checks that the full chain:
#   parse → preprocess → extract → score → report
# produces correct end-to-end results.
#
# These tests use real files on disk (via tmp_path fixtures) and run the
# actual pipeline modules together, just as main.py does.
# =============================================================================

import json
import csv
import pytest

from parser.resume_parser import ResumeParser
from parser.jd_parser import JobDescriptionParser
from nlp.preprocessor import TextPreprocessor
from nlp.extractor import ResumeFeatureExtractor
from scoring.scorer import ResumeScorer
from reports.reporter import ReportGenerator
from main import process_resumes, sort_results, load_job_description


# =============================================================================
# INTEGRATION FIXTURES
# =============================================================================

@pytest.fixture
def pipeline(nlp_model, tmp_path):
    """
    Build and return all pipeline components as a dict.
 
    This fixture creates one instance of each module for use across all
    integration tests in this file. By using scope="function" (the default),
    each test gets a fresh set of instances — preventing state leakage
    between tests.
    """

    return {
        "resume_parser": ResumeParser(),
        "jd_parser": JobDescriptionParser(),
        "preprocessor": TextPreprocessor(nlp_model),
        "extractor": ResumeFeatureExtractor(nlp_model),
        "scorer": ResumeScorer(),
        "reporter": ReportGenerator(output_dir=str(tmp_path/ "output"))  
    }

@pytest.fixture
def populated_folder(tmp_path, sample_jd_text):
    """
    Create a folder with 3 distinct CVs and a JD file.
 
    The three CVs have clearly different quality levels so IT-01 can
    verify that the ranking order is correct.
    """

    strong_cv = """
    Alice Senior — Lead Python Developer
    Experience at TechCorp (2018 - 2024).
    I have 6 years of experience in backend development.
    Skills: Python, FastAPI, Docker, PostgreSQL, AWS, Git.
    MSc Computer Science 2017.
    """

    average_cv = """
    Bob Mid — Python Developer
    Developer at SmallCo (2021 - 2024).
    Skills: Python, Docker, Git.
    Bachelor in Computer Science 2020.
    """

    weak_cv = """
    Carol Junior — Marketing Assistant
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

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullPipelineRanking:

    def test_it01_strong_candidate_ranks_above_weak(self, pipeline, populated_folder):
        """
        IT-01: Running the pipeline with a strong, average, and weak candidate
        produces a ranking where strong > average > weak by total_score.
        """
        p= pipeline
        f= populated_folder

        # Parse JD
        jd_criteria = load_job_description(
            str(f["jd_file"]), p["jd_parser"]
        )
        assert jd_criteria is not None

        # Run the full pipeline for all CVs
        scored_results, failed = process_resumes(
            folder_path=str(f["cv_folder"]),
            jd_criteria=jd_criteria,
            resume_parser=p["resume_parser"],
            preprocessor=p["preprocessor"],
            extractor=p["extractor"],
            scorer=p["scorer"],
        )

        # All three CVs should be processed successfully
        assert len(failed) == 0
        assert len(scored_results) == 3

        # Sort results as main.py would
        ranked = sort_results(scored_results)

        # The strong candidate (alice) must socre higher than the weak candidate (carol)
        scores = {r["name"]: r["total_score"] for r in ranked}
        assert scores["alice"] > scores["carol"]

        # The ranking must be correct: Alice is first, carol is last
        assert ranked[0]["name"] == "alice"
        assert ranked[-1]["name"] == "carol"

class TestCSVOutput:

    def test_it02_csv_has_header_plus_one_row_per_candidate(self, pipeline, populated_folder):
        """
        IT-02: The CSV file has exactly N+1 rows (1 header + N candidates).
        """
        p = pipeline
        f = populated_folder

        jd_criteria = load_job_description(str("jd_file"), p["jd_parser"])
        scored_results, _ =process_resumes(
            str(f["cv_folder"]), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )
        ranked = sort_results(scored_results)

        # Generate the CSV report
        csv_path = p["reporter"].save_csv(ranked)

        # Count rows in the CSV file
        with open(csv_path, encodinh="utf-8-sig") as f_csv:
            reader = csv.reader(f_csv)
            rows = list(reader)

        # rows[0] is the header; rows[1:] are the candidartes
        assert len(rowns) == len(ranked) + 1

class TestJSONOutput:

    def test_it03_json_is_parseable_and_has_expected_structure(self, pipeline, populated_folder):
        """
        IT-03: The JSON file parses without errors and contains
        'metadata' and 'candidates' keys with the correct content.
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

        # Parse the JSON file
        with open(json_path, encoding="utf-8") as f_json:
            data = json.load(f_json)

        # Top- level structure
        assert "metadata" in data
        assert "candidates" in data

        # Metadata content
        assert data["metadata"]["total_candidates"] == len(ranked)

        # Each candidate entry must have all required fields
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
        IT-04: All total_score values in the JSON output are between 0 and 100.
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
        IT-04: Including an invalid file (too short to pass validation)
        in the CV folder does not prevent valid CVs from being processed.
        Valid CVs are scored; the invalid file appears in failed_files.
        """

        p = pipeline

        cv_folder = tmp_path / "resumes"
        cv_folder.mkdir()

        # One valid CV that should be processed successfully    
        (cv_folder / "valid_candidate.txt").write_text(strong_cv_text, encoding="utf-8")

        # One file that is too short to pass minimum lenght validation
        (cv_folder / "too_short.txt").write_text("Hi.", encoding="utf-8")

        # One file with unsurpported extension - silently skipped by the system
        (cv_folder / "notes.xyz").write_text("Meeting notes.", encoding="utf-8")

        jd_file = tmp_path / "jd.txt"
        jd_file.write_text(sample_jd_text, encoding="utf-8")

        jd_criteria = load_job_description(str(jd_file), p["jd_parser"])
        scored_results, failed_files = process_resumes(
            str(cv_folder), jd_criteria,
            p["resume_parser"], p["preprocessor"], p["extractor"], p["scorer"]
        )

        # The valid Cv must be scored
        assert len(scored_results) == 1
        assert scored_results[0]["name"] == "valid_candidate"

        # the too-short file must appear in failed_files (not silently dropped)
        assert len(failed_files) == 1
        assert failed_files[0]["name"] == "too short"

class TestSortOrder:

    def test_sort_by_score_descending(self):
        results = [
            {"name": "a", "total_score": 45.0},
            {"name": "b", "total_score": 80.0},
            {"name": "c", "total_score": 60.0},
        ]

        ranked = sort_results(results)

        assert ranked[0]["total_scores"] == 80.0
        assert ranked[1]["total_scores"] == 60.0
        assert ranked[2]["total_scores"] == 45.0

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
