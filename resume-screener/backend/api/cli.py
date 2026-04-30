# =============================================================================
# main.py - Orquestrador do pipeline e entrada CLI
# =============================================================================
# Responsável por ler argumentos da linha de comandos, inicializar os módulos
# e executar o fluxo completo: vaga -> CVs -> extração -> pontuação -> relatórios.
# A lógica de negócio fica nos módulos próprios; este ficheiro só coordena a ordem.
# =============================================================================

import argparse
import logging
import sys
import spacy

from pathlib import Path
from backend.api.scoring_config import TEXT_CONFIG
from backend.parser.resume_parser import ResumeParser
from backend.parser.jd_parser import JobDescriptionParser
from backend.nlp.preprocessor import TextPreprocessor
from backend.nlp.extractor import ResumeFeatureExtractor
from backend.scoring.scorer import ResumeScorer
from backend.reports.reporter import ReportGenerator

# Configuração global de logging herdada pelos módulos via logging.getLogger(__name__).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Cria o parser de argumentos da CLI.

    Define caminhos de entrada e saída com valores por defeito para permitir
    executar o pipeline sem argumentos adicionais.
    """
    parser = argparse.ArgumentParser(
        prog="resume_screener",
        description=(
            "Automated Resume Screener â€”"
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


def load_spacy_model() -> object:
    """
    Carrega uma única instância do modelo spaCy configurado.

    A mesma instância é injetada nos módulos de NLP para evitar carregamentos
    repetidos do modelo.
    """

    model_name= TEXT_CONFIG["spacy_model"]
    logger.info(f"Loading spaCy model '{model_name}'...")

    try:
        nlp = spacy.load(model_name)
        logger.info("spaCy model loaded successfully.")
        return nlp

    except OSError:
        # Mostra uma mensagem acionável quando o pacote do modelo não está instalado.
        logger.error(
            f"spaCy model '{model_name}' not found. Please install it using:\n"
            f"    python -m spacy download {model_name}"
        )
        sys.exit(1)


def load_job_description(jd_path: str, jd_parser: JobDescriptionParser) -> dict | None:
    """
    Lê o ficheiro da vaga e devolve os critérios extraídos.
    """
    path = Path(jd_path)

    if not path.exists():
        logger.error(f"Job Description file not found: {jd_path}")
        return None

    try:
        jd_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Usa Latin-1 como fallback para ficheiros que não estejam em UTF-8.
        jd_text = path.read_text(encoding="latin-1")

    logger.info(f"Job Description loaded: {path.name} ({len(jd_text)} characters)")
    return jd_parser.parse(jd_text)


def process_resumes(folder_path: str, jd_criteria: dict, resume_parser: ResumeParser, preprocessor: TextPreprocessor, extractor: ResumeFeatureExtractor, scorer: ResumeScorer) -> tuple[list[dict], list[dict]]:
    """
    Executa parsing, pré-processamento, extração e pontuação para todos os CVs.

    Ficheiros inválidos são guardados em failed_files para relatório posterior,
    sem interromper o processamento dos restantes candidatos.
    """
    parse_results = resume_parser.parse_folder(folder_path)

    if not parse_results:
        logger.warning(f"No supported CV files found in folder: {folder_path}")
        return [], []

    scored_results: list[dict] = []
    failed_files: list[dict] = []

    for parse_result in parse_results:
        candidate_name = parse_result["name"]

        # Ignora CVs que falharam no parsing ou ficaram sem texto extraído.
        if parse_result["error"] is not None or not parse_result["text"]:
            logger.warning(f"Skipping '{candidate_name}': {parse_result.get('error', 'empty text')}")
            failed_files.append(parse_result)
            continue

        logger.info(f"Processing: {candidate_name}")

        preprocessed = preprocessor.process(parse_result["text"])

        # Procura apenas competências pedidas pela vaga.
        features = extractor.extract(
            resume_text=preprocessed["clean_text"],
            jd_skills=jd_criteria.get("skills", [])
        )

        # Usa os tokens do pré-processador como base principal para TF-IDF.
        features["keywords"] = preprocessed["tokens"]

        score_result= scorer.score(
            resume_features=features,
            jd_criteria=jd_criteria
        )

        # Acrescenta o nome do candidato para os relatórios.
        score_result["name"] = candidate_name

        scored_results.append(score_result)

    return scored_results, failed_files


def sort_results(results: list[dict]) ->list[dict]:
    """
    Ordena por pontuação descendente e, em empate, por nome ascendente.
    """

    return sorted(
        results,
        key=lambda x: (-x["total_score"], x["name"].lower())
    )


def report_failures(failed_files: list[dict]) -> None:
    """
    Regista no log os CVs que não puderam ser processados.
    """

    if not failed_files:
        return

    logger.warning(f"{len(failed_files)} file(s) could not be processed:")
    for failed_file in failed_files:
        logger.warning(f" - {failed_file['name']}: {failed_file.get('error', 'unknown error')}")


def main() -> None:
    """
    Coordena a execução completa do resume screener.
    """

    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    logger.info("=" * 60)
    logger.info("AUTOMATED RESUME SCREENER â€” Starting")
    logger.info(f"  Job Description: {args.jd}")
    logger.info(f"  Resumes:         {args.resumes}")
    logger.info(f"  Output:          {args.output}")
    logger.info("=" * 60)

    nlp = load_spacy_model()

    resume_parser = ResumeParser()
    jd_parser = JobDescriptionParser(nlp)
    preprocessor = TextPreprocessor(nlp)
    extractor = ResumeFeatureExtractor(nlp)
    scorer = ResumeScorer()
    reporter = ReportGenerator(output_dir=args.output)

    jd_criteria = load_job_description(args.jd, jd_parser)

    if jd_criteria is None:
        logger.error("Cannot proceed without a valid Job Description. Exiting.")
        sys.exit(1)

    scored_results, failed_files = process_resumes(
        folder_path=args.resumes,
        jd_criteria=jd_criteria,
        resume_parser=resume_parser,
        preprocessor=preprocessor,
        extractor=extractor,
        scorer=scorer
    )

    # Sai de forma controlada se nenhum CV válido foi processado.
    if not scored_results:
        logger.error("No candidaes were sucessfully processed. No reports generated.")
        report_failures(failed_files)
        sys.exit(1)

    ranked_results = sort_results(scored_results)

    reporter.print_terminal(ranked_results)

    csv_path = reporter.save_csv(ranked_results)
    json_path = reporter.save_json(ranked_results)
    txt_path = reporter.save_txt(ranked_results)

    logger.info(f"Reports saved to: {args.output}/")
    logger.info(f"  CSV:  {csv_path.name}")
    logger.info(f"  JSON: {json_path.name}")
    logger.info(f"  TXT:  {txt_path.name}")

    report_failures(failed_files)

    logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    main()
