# =============================================================================
# main.py — Pipeline Orchestrator and CLI Entry Point
# =============================================================================
# Responsibility: Parse CLI arguments, initialise all modules, and execute
# the full screening pipeline from file parsing to report generation.
#
# This module is intentionally thin — it contains no business logic of its own.
# All logic lives in the dedicated modules. main.py only knows the order in
# which to call them.
#
# Pipeline execution order:
#   1. Load spaCy model ONCE (shared across all NLP modules)
#   2. Parse the Job Description → extract JD criteria
#   3. Parse all CV files in the resumes folder → extract raw text
#   4. For each valid CV:
#        a. Preprocess the text → clean text + tokens
#        b. Extract features    → skills, experience, education, keywords
#        c. Score the candidate → weighted score + classification
#   5. Sort results by score (desc), then name (asc) on ties
#   6. Generate all reports (terminal, CSV, JSON, TXT)
#
# Usage:
#   python main.py                              # uses defaults
#   python main.py --jd path/to/jd.txt --resumes ./cvs --output ./results
# =============================================================================

import argparse
import logging
import sys
import spacy

from pathlib import Path
from config import TEXT_CONFIG
from parser.resume_parser import ResumeParser
from parser.jd_parser import JobDescriptionParser
from nlp.preprocessor import TextPreprocessor
from nlp.extractor import ResumeFeatureExtractor
from scoring.scorer import ResumeScorer
from reports.reporter import ReportGenerator

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Configure the root logger for the entire application.
# All modules use logging.getLogger(__name__) which creates child loggers
# that inherit this root configuration automatically.
#
# Format explanation:
#   %(asctime)s     — timestamp of the log message
#   %(levelname)-8s — level name padded to 8 chars (INFO, WARNING, ERROR)
#   %(name)s        — the module that generated the message (e.g. nlp.extractor)
#   %(message)s     — the log message itself
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# CLI ARGUMENT PARSING
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.
 
    Defines the three arguments that control the pipeline's input and output
    paths. All three have sensible defaults so the tool works out-of-the-box
    without any arguments, as documented in SDLC.
 
    Returns:
        argparse.ArgumentParser: Configured parser ready to call .parse_args().
    """
    parser = argparse.ArgumentParser(
        prog="resume_screener",
        description=(
            "Automated Resume Screener —"
            "Rank candidates against a Job Description using NLP. "
        ),
    )

    parser.add_argument(
        "--jd",
        type=str,
        default="job_description.txt",
        help="Path to the Job Description text file. Default: ./job_description.txt",
    )

    parser.add_argument(
        "--resumes",
        type=str,
        default="./resumes",
        help="Path to the folder containing candidate CV files. Default: ./resumes",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Path to the output folder for generated reports. Default: ./output",
    )
    return parser

# =============================================================================
# PIPELINE STEPS - individual functions for clarity and testability
# =============================================================================

def load_spacy_model() -> object:
    """
    Load the spaCy language model defined in config.py.
 
    This is called ONCE at startup and the returned model is shared across
    all NLP modules (Dependency Injection pattern). Loading the model takes
    ~0.5s and uses ~15MB of RAM — doing it once is critical for performance
    when processing large batches (NFR-01).
 
    Returns:
        spacy.Language: The loaded spaCy model.
 
    Raises:
        SystemExit: If the model is not installed, logs a clear error with
                    the installation command and exits with code 1.
    """

    model_name= TEXT_CONFIG["spacy_model"]
    logger.info(f"Loading spaCy model '{model_name}'...")

    try:
        nlp = spacy.load(model_name)
        logger.info("spaCy model loaded successfully.")
        return nlp
    
    except OSError:
        # This error means the model package is not installed.
        # We catch it here and provide a clear, actionable error message
        # rather than letting a cryptic OSError propagate to the user.
        logger.error(
            f"spaCy model '{model_name}' not found. Please install it using:\n"
            f"    python -m spacy download {model_name}"
        )
        sys.exit(1)

def load_job_description(jd_path: str, jd_parser: JobDescriptionParser) -> dict | None:
    """
    Read and parse the Job Description file.
 
    Args:
        jd_path (str): Path to the JD text file.
        jd_parser (JobDescriptionParser): Initialised JD parser instance.
 
    Returns:
        dict | None: Parsed JD criteria dict, or None if the file cannot be read.
    """
    path = Path(jd_path)

    if not path.exists():
        logger.error(f"Job Description file not found: {jd_path}")
        return None
    
    # Read the JD file content
    try:
        jd_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to Latin-1 for files not saved as UTF-8 (same strategy as
        # resume_parser._parse_txt() — consistent encoding handling throughout)
        jd_text = path.read_text(encoding="latin-1")

    logger.info(f"Job Description loaded: {path.name} ({len(jd_text)} characters)")
    return jd_parser.parse(jd_text)


def process_resumes(folder_path: str, jd_criteria: dict, resume_parser: ResumeParser, preprocessor: TextPreprocessor, extractor: ResumeFeatureExtractor, scorer: ResumeScorer) -> tuple[list[dict], list[dict]]:
    """
    Run the full extraction and scoring pipeline for all CVs in a folder.
 
    Processes each CV file through parsing → preprocessing → extraction → scoring.
    CVs that fail to parse are collected separately and reported at the end
    without aborting the batch (NFR-06: Robustness).
 
    Args:
        folder_path (str): Path to the folder containing CV files.
        jd_criteria (dict): Parsed JD criteria from load_job_description().
        resume_parser (ResumeParser): Initialised resume parser.
        preprocessor (TextPreprocessor): Initialised text preprocessor.
        extractor (ResumeFeatureExtractor): Initialised feature extractor.
        scorer (ResumeScorer): Initialised scoring engine.
 
    Returns:
        tuple[list[dict], list[dict]]:
            - scored_results: List of score dicts (one per valid CV).
            - failed_files:   List of dicts for CVs that failed to parse.
    """
    # Step 1: Parse all CV files in the folder.
    # parse_folder() returns a result dict for every supported file found,
    # with error=None for successes and error=<message> for failures.
    parse_results = resume_parser.parse_folder(folder_path)

    if not parse_results:
        logger.warning(f"No supported CV files found in folder: {folder_path}")
        return [], []
    
    scored_results: list[dict] = []
    failed_files: list[dict] = []

    for parse_result in parse_results:
        candidate_name = parse_result["name"]

        # --- Separate failed parses from successful ones ---
        # A CV with a non-None error OR empty text is not processable.
        # We collect it for the failure report and skip to the next file.
        if parse_result["error"] is not None or not parse_result["text"]:
            logger.warning(f"Skipping '{candidate_name}': {parse_result.get('error', 'empty text')}")
            failed_files.append(parse_result)
            continue

        logger.info(f"Processing: {candidate_name}")

        # Step 2: Preprocess the extracted text.
        # Produces clean_text (for regex operations) and tokens (for TF-IDF).
        preprocessed = preprocessor.process(parse_result["text"])

        # Step 3: Extract features from the preprocessed CV.
        # We pass jd_criteria["skills"] so the extractor knows which skills
        # to look for — it only reports skills the JD actually requires.
        features = extractor.extract(
            resume_text=preprocessed["clean_text"],
            jd_skills=jd_criteria.get("skills", [])
        )

        # Override the keyword list with the preprocessor's tokens.
        # The preprocessor's token list is more complete than the extractor's
        # because it runs on the full clean text. The extractor's keywords
        # are used for NER context; the preprocessor's tokens feed TF-IDF.
        features["keywords"] = preprocessed["tokens"]

        # Step 4: Score the candidates
        score_result= scorer.score(
            resume_features=features,
            jd_criteria=jd_criteria
        )

        # Augment the score result with the candidate's name so the reporter
        # can display it without needing access to the original parse result.
        score_result["name"] = candidate_name
        
        scored_results.append(score_result)

    return scored_results, failed_files

def sort_results(results: list[dict]) ->list[dict]:
    """
    Sort scored results by total_score descending, then name ascending on ties.
 
    The secondary sort by name implements the business rule from SDLC:
    "In the event of a score tie, candidates are presented in alphabetical
    order by filename."
 
    Python's sort is stable, so we can achieve this with a tuple key:
    (-score, name) sorts by score descending (negative for reverse) and
    name ascending for ties.
 
    Args:
        results (list[dict]): Unsorted list of scored candidate dicts.
 
    Returns:
        list[dict]: New list sorted by score desc, name asc on ties.
    """

    return sorted(
        results,
        key=lambda x: (-x["total_score"], x["name"].lower())
    )

def report_failures(failed_files: list[dict]) -> None:
    """
    Log a summary of all CV files that failed to parse.
 
    Called after all reports are generated so failures are visible but do
    not interrupt the main output. Each failure shows the filename and the
    reason it was skipped.
 
    Args:
        failed_files (list[dict]): List of failed parse result dicts.
    """

    if not failed_files:
        return
    
    logger.warning(f"{len(failed_files)} file(s) could not be processed:")
    for failed_file in failed_files:
        logger.warning(f" - {failed_file['name']}: {failed_file.get('error', 'unknown error')}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Orchestrate the full resume screening pipeline.
 
    Reads CLI arguments, initialises all modules (loading spaCy once),
    processes the JD and all CVs, scores and ranks candidates, and
    generates all four output reports.
    """

    # --- Parse CLI arguments ---
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    logger.info("=" * 60)
    logger.info("AUTOMATED RESUME SCREENER — Starting")
    logger.info(f"  Job Description: {args.jd}")
    logger.info(f"  Resumes:         {args.resumes}")
    logger.info(f"  Output:          {args.output}")
    logger.info("=" * 60)

    # --- Load spaCy model ONCE and share across modules ---
    nlp = load_spacy_model()

    # --- Initialise all modules with the shared spaCy model ---
    resume_parser = ResumeParser()
    jd_parser = JobDescriptionParser(nlp)
    preprocessor = TextPreprocessor(nlp)
    extractor = ResumeFeatureExtractor(nlp)
    scorer = ResumeScorer()
    reporter = ReportGenerator()

    # --- Parse and process the Job Description ---
    jd_criteria = load_job_description(args.jd, jd_parser)

    if jd_criteria is None:
        logger.error("Cannot proceed without a valid Job Description. Exiting.")
        sys.exit(1)

    # --- Process all CVs and score them ---
    scored_results, failed_files = process_resumes(
        folder_path=args.resumes,
        jd_criteria=jd_criteria,
        resume_parser=resume_parser,
        preprocessor=preprocessor,
        extractor=extractor,
        scorer=scorer
    )

    # Exit grecefully if no valid CVs were processed
    if not scored_results:
        logger.error("No candidaes were sucessfully processed. No reports generated.")
        report_failures(failed_files)
        sys.exit(1)

    # --- Sort results by score desc, then name asc on ties ---
    ranked_results = sort_results(scored_results)

    # --- Generate all reports ---
    reporter.print_terminal(ranked_results)

    csv_path = reporter.save_csv(ranked_results)
    json_path = reporter.save_json(ranked_results)
    txt_path = reporter.save_txt(ranked_results)

    logger.info(f"Reports saved to: {args.output}/")
    logger.info(f"  CSV:  {csv_path.name}")
    logger.info(f"  JSON: {json_path.name}")
    logger.info(f"  TXT:  {txt_path.name}")

    # --- Report any files that failed to parse ---
    report_failures(failed_files)

    logger.info("Pipeline execution completed.")

if __name__ == "__main__":
    main() 