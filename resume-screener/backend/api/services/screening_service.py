"""
Serviço de orquestração do pipeline de screening.
 
Integra o pipeline v1.0 (parser, NLP, scoring) com a camada de persistência (DB, ORM).
Executa em background via FastAPI BackgroundTasks para não bloquear requisições HTTP.
 
Princípios:
- Robusto: falha de um candidato não bloqueia outros
- Transparente: erros registados em ProcessingError
- Auditável: logging estruturado de cada etapa
- Seguro: sem exposição de dados sensíveis em logs
"""

import logging
from typing import Any
from datetime import datetime

from backend.parser.jd_parser import JobDescriptionParser
from backend.nlp.preprocessor import TextPreprocessor
from backend.nlp.extractor import ResumeFeatureExtractor
from backend.scoring.scorer import ResumeScorer

from backend.api.db.models import (
  Candidate,
  Result,
  ProcessingError,
  ProcessStatus,
)
from backend.api.services.process_service import ProcessService

logger = logging.getLogger(__name__)
class ScreeningService:
    """
    Orquestra a execução do pipeline de screening v1.0.
 
    Responsabilidades:
    - Carregar e validar estado do processo
    - Extrair critérios da JD
    - Processar cada CV individualmente
    - Persistir resultados
    - Gerir transições de estado
 
    Design: composição com v1.0 modules injetados. Não herança.
    """

    def __init__(self, nlp: Any) -> None:
        """
          Inicializa ScreeningService com componentes v1.0.
  
          Args:
              nlp: spacy.Language carregado uma única vez no startup.
                  Injetado aqui para evitar carregar múltiplas vezes.
          """
        self.nlp = nlp
        self._jd_parser = JobDescriptionParser(nlp)
        self._preprocessor = TextPreprocessor(nlp)
        self._extractor = ResumeFeatureExtractor(nlp)
        self._scorer = ResumeScorer()

        logger.debug("ScreeningService initialised with v1.0 components.")

    def run(self, process_id: str, db_session: Any) -> None:
        """
          Executa o screening completo de um processo em background.
  
          Fluxo:
          1. Load processo e validar estado (files_uploaded)
          2. Mark processo como processing
          3. Load critérios da JD
          4. Para cada candidato: extract features → score → persist
          5. Mark processo como completed (ou failed se erro fatal)
  
          Erro de um candidato não bloqueia outros (robustez).
          Erro fatal (DB, JD inválida) marca processo como failed.
  
          Args:
              process_id: UUID do processo
              db_session: SQLAlchemy session para operações DB
  
          Executada via BackgroundTasks, sem retorno ao cliente HTTP.
        """

        process_service = ProcessService(db_session)

        try:
            logger.info("Starting screening process for process: %s", process_id)

            # 1. Load e validar processo
            process = process_service.get_process(process_id)

            if process.status != ProcessStatus.FILES_UPLOADED:
                # Estado inválido para screening
                logger.warning(
                  "Cannot run screening on process %s in state %s, requires files_uploaded.",
                  process_id,
                  process.status,
                )
                process_service.mark_failed(
                  process_id,
                  error_message="Cannot run screening: process not in files_uploaded state.",
                )
                return

            # 2. Mark como processando
            process_service.mark_processing(process_id)
            logger.info("Process %s marked as processing", process_id)

            # 3. Load JD e extrai critérios
            jd_criteria = self._load_jd_criteria(process.jd_text)
            logger.info(
              "JD criteria loaded for process %s - skills: %d, min_exp: %d years",
              process_id,
              len(jd_criteria.get("skills", [])),
              jd_criteria["min_experience"],
            )

            # 4. Load candidatos e processa cada um
            candidates = db_session.query(Candidate).filter(
              Candidate.process_id == process_id,
            ).all()

            if not candidates:
                logger.warning("No candidates found for process %s", process_id)
                process_service.mark_completed(process_id)
                return

            logger.info(
              "Processing %d candidate(s) for process %s",
              len(candidates),
              process_id,
            )

            # Processa cada candidato individualmente
            for candidate in candidates:
                try:
                    # Pipeline v1.0: preprocess -> extract -> score
                    result = self._process_candidate(candidate, jd_criteria)

                    # Persist resultado
                    db_session.add(result)

                    logger.debug(
                      "Candidate %s (id=%s) scored: %.1f (%s)",
                      candidate.name,
                      candidate.id,
                      result.total_score,
                      result.category,
                    )

                except Exception as e: # pylint: disable=broad-exception-caught
                    # Falha de um candidato não bloqueia outros
                    logger.error(
                      "Failed to process candidate %s (id=%s): %s",
                      candidate.name,
                      candidate.id,
                      str(e),
                    )
                    # Regista erro na base de dados para auditoria
                    error_record = ProcessingError(
                      process_id=process_id,
                      candidate_id=candidate.id,
                      stage="scoring",
                      message=f"{type(e).__name__}: {str(e)}",
                    )
                    db_session.add(error_record)

                # Commit de todos os resultados e erros
                db_session.commit()

            # 5. Mark processo como completed
            process_service.mark_completed(process_id)
            logger.info("Screening completed successfully for process %s", process_id)
        except Exception as e: # pylint: disable=broad-exception-caught
            # Erro fatal -> marca processo como failed
            logger.error(
              "Fatal error during screening process %s: %s",
              process_id,
              str(e),
            )
            db_session.rollback()

            try:
                process_service.mark_failed(
                  process_id,
                  error_message=f"Screening failed: {str(e)}")
            except Exception as e2: # pylint: disable=broad-exception-caught
                logger.error(
                  "Could not mark process %s as failed: %s",
                  process_id,
                  str(e2),
                )

    def get_results(self, process_id: str, db_session: Any) -> dict:
        """
          Retorna resultados do screening para um processo.
  
          Comportamento por estado:
          - processing: {status: "processing"} — caller retorna 202
          - completed: {status: "completed", summary: {...}, candidates: [...]}
          - failed: {status: "failed", error_message: "..."}
          - qualquer outro: {status: "...", message: "..."}
  
          Args:
              process_id: UUID do processo
              db_session: SQLAlchemy session
  
          Returns:
              dict com estrutura esperada por GET /results route
          """

        process_service = ProcessService(db_session)

        # Load processo
        process = process_service.get_process(process_id)

        # Estado: ainda processando
        if process.status == ProcessStatus.PROCESSING:
            logger.debug("Process %s is still processing", process_id)
            return {"status": "processing"}

        # Estado: failed
        if process.status == ProcessStatus.FAILED:
            logger.debug("Process %s failed: %s", process_id, process.error_message)
            return {
              "status": "failed",
              "error_message": process.error_message or "Unknown error",
            }

        # Estado: completed
        if process.status == ProcessStatus.COMPLETED:
            # Load resultados ordenados pelo score descendente
            results = (
              db_session.query(Result)
              .filter(
                Result.candidate_id.in_(
                  db_session.query(Candidate).filter(
                    Candidate.process_id == process_id,
                  )
                )
              )
            .order_by(Result.total_score.desc())
            .all()
          )

            # Constrói summary
            summary = {
              "total": len(results),
              "strong_matches": sum(
                1 for r in results if r.category == "Strong match"
              ),
              "potential_matches": sum(
                1 for r in results if r.category == "Potential match"
              ),
              "weak_matches": sum(
                1 for r in results if r.category == "Weak match"
              ),
            }

            # Constrói array de candidatos com ranking
            candidates_response = []
            for rank, result in enumerate(results, start=1):
                candidate = db_session.query(Candidate).filter(
                  Candidate.id == result.candidate_id,
                ).first()

                candidates_response.append({
                  "rank": rank,
                  "name": candidate.name if candidate else "Unknown",
                  "total_score": result.total_score,
                  "category": result.category,
                  "breakdown": result.breakdown,
                  "matched_skills": result.matched_skills,
                  "required_skills": result.required_skills,
                  "missing_skills": result.missing_skills,
                  "experience_years_found": result.experience_years_found,
                })

            logger.info(
              "Results retrieved for process %s: %d candidates, %d strong_matches",
              process_id,
              summary["total"],
              summary["strong_matches"],
            )

            return {
              "status": "completed",
              "summary": summary,
              "candidates": candidates_response,
            }

        # Estado inesperado (created, files_uploaded, cancelled)
        logger.warning(
          "Process %s in unexpected state %s. No results available.",
          process_id,
          process.status,
        )
        return {
          "status": process.status,
          "message": "No results available for this process state.",
        }

    def _load_jd_criteria(self, jd_text: str) -> dict:
        """
        Extrai critérios de avaliação da JD usando v1.0 parser.
 
        Wrapper sobre JobDescriptionParser que garante consistência
        de formato esperado pelo resto do pipeline.
 
        Args:
            jd_text: Texto bruto da descrição de vaga
 
        Returns:
            dict com chaves: skills, min_experience, education_level, keywords, raw_text
        """

        criteria = self._jd_parser.parse(jd_text)

        logger.debug(
          "JD parsed: %s skills, %d keywords, min_exp=%d years, education=%s",
          len(criteria["skills"]),
          len(criteria["keywords"]),
          criteria["min_experience"],
          criteria["education_level"],
        )

        return criteria

    def _process_candidate(self, candidate: Candidate, jd_criteria: dict) -> Result:
        """
        Processa um candidato individual através do pipeline v1.0.
 
        Pipeline (3 etapas):
        1. Preprocess: limpeza de texto + tokenização
        2. Extract: features (skills, exp, edu, keywords)
        3. Score: cálculo de pontuação ponderada
 
        A ordem é importante: extractor recebe texto limpo (melhor acurácia),
        e tokens lematizados do preprocessor (mais robustos que os do extractor).
 
        Args:
            candidate: Modelo ORM Candidate (com raw_text preenchido)
            jd_criteria: dict com skills, min_experience, education_level, keywords
 
        Returns:
            Modelo ORM Result (preenchido, não persistido)
 
        Raises:
            Exception: Qualquer erro em parsing, NLP ou scoring.
                      Será capturado por run() e registado em ProcessingError.
        """

        # 1. Preprocessing: limpeza e tokenização
        preprocessed = self._preprocessor.process(candidate.raw_text)

        # 2. Features extraction: features estruturadas do CV
        features = self._extractor.extract(
          resume_text=preprocessed["clean_text"],
          jd_skills=jd_criteria["skills"],
        )

        # Override keywords com tokens lematizados do preprocessor
        features["keywords"] = preprocessed["tokens"]

        # 3. Score: Pontuação ponderada contra critérios da JD
        score_result = self._scorer.score(
          resume_features=features,
          jd_criteria=jd_criteria,
        )

        # 4. Construir o modelo ORM Result com os dados do score
        result = Result(
          candidate_id=candidate.id,
          total_score=score_result["total_score"],
          category=score_result["category"],
          breakdown=score_result["breakdown"],
          matched_skills=score_result["matched_skills"],
          required_skills=jd_criteria["skills"],
          missing_skills=score_result["missing_skills"],
          experience_years_found=score_result["experience_years_found"],
          created_at=datetime.utcnow(),
        )

        logger.debug(
          "Candidate %s processed: score=%.1f, category=%s, skills=%d/%d",
          candidate.name,
          result.total_score,
          result.category,
          len(result.matched_skills),
          len(result.matched_skills) + len(score_result["missing_skills"]),
        )

        return result
